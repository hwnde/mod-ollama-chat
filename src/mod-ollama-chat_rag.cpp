#include "mod-ollama-chat_rag.h"
#include "mod-ollama-chat_config.h"
#include "Log.h"
#include "Random.h"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace fs = std::filesystem;

OllamaRAGSystem::OllamaRAGSystem() : m_initialized(false) {}

OllamaRAGSystem::~OllamaRAGSystem() {}

bool OllamaRAGSystem::Initialize()
{
    if (m_initialized) {
        return true;
    }

    m_ragEntries.clear();
    m_idIndex.clear();
    m_vocabulary.clear();

    // Use the configured RAG data path directly
    std::string fullPath = g_RAGDataPath;

    if (!LoadRAGDataFromDirectory(fullPath)) {
        LOG_ERROR("server.loading", "[Ollama Chat RAG] Failed to load RAG data from directory: {}", fullPath);
        return false;
    }

    // Build vocabulary from all entries
    std::unordered_set<std::string> vocabSet;
    for (const auto& entry : m_ragEntries) {
        auto tokens = TokenizeText(PreprocessText(entry.title + " " + entry.content));
        for (const auto& token : tokens) {
            vocabSet.insert(token);
        }
        for (const auto& keyword : entry.keywords) {
            auto keywordTokens = TokenizeText(PreprocessText(keyword));
            for (const auto& token : keywordTokens) {
                vocabSet.insert(token);
            }
        }
    }
    m_vocabulary.assign(vocabSet.begin(), vocabSet.end());

    // Build id -> entry index for one-hop reference resolution.
    // m_ragEntries is never mutated after this point, so element pointers stay valid.
    for (const auto& entry : m_ragEntries) {
        m_idIndex[entry.id] = &entry;
    }

    // Validate references at load. Logs only; never aborts startup.
    uint32_t refTotal = 0;
    uint32_t refDangling = 0;
    for (const auto& entry : m_ragEntries) {
        for (const auto& refId : entry.references) {
            ++refTotal;
            auto it = m_idIndex.find(refId);
            if (it == m_idIndex.end()) {
                ++refDangling;
                LOG_ERROR("server.loading",
                          "[Ollama Chat RAG] Entry '{}' references unknown id '{}'",
                          entry.id, refId);
            } else if (it->second->short_description.empty()) {
                LOG_WARN("server.loading",
                         "[Ollama Chat RAG] Reference target '{}' has no short_description; "
                         "will render title only", refId);
            }
        }
    }
    LOG_INFO("server.loading", "[Ollama Chat RAG] References checked: {}, dangling: {}",
             refTotal, refDangling);

    if (g_RAGImprovedScoring)
    {
        BuildIdf();
        BuildEntryVectors();
        LOG_INFO("server.loading", "[Ollama Chat RAG] IDF table built ({} terms)", m_idf.size());
    }

    m_initialized = true;
    LOG_INFO("server.loading", "[Ollama Chat RAG] Initialized with {} entries and {} vocabulary terms",
             m_ragEntries.size(), m_vocabulary.size());

    return true;
}

bool OllamaRAGSystem::LoadRAGDataFromDirectory(const std::string& directoryPath)
{
    try {
        if (!fs::exists(directoryPath)) {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] Directory does not exist: {}", directoryPath);
            return false;
        }

        if (!fs::is_directory(directoryPath)) {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] Path is not a directory: {}", directoryPath);
            return false;
        }

        uint32_t loadedFiles = 0;
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".json") {
                if (LoadRAGDataFromFile(entry.path().string())) {
                    loadedFiles++;
                }
            }
        }

        LOG_INFO("server.loading", "[Ollama Chat RAG] Loaded {} JSON files from {}", loadedFiles, directoryPath);
        return loadedFiles > 0;
    }
    catch (const std::exception& e) {
        LOG_ERROR("server.loading", "[Ollama Chat RAG] Error loading directory {}: {}", directoryPath, e.what());
        return false;
    }
}

bool OllamaRAGSystem::LoadRAGDataFromFile(const std::string& filePath)
{
    try {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] Cannot open file: {}", filePath);
            return false;
        }

        nlohmann::json jsonData;
        file >> jsonData;

        if (!jsonData.is_array()) {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] JSON file must contain an array of entries: {}", filePath);
            return false;
        }

        uint32_t entriesLoaded = 0;
        for (const auto& item : jsonData) {
            try {
                RAGEntry entry;
                entry.id = item.value("id", "");
                entry.title = item.value("title", "");
                entry.short_description = item.value("short_description", "");
                entry.content = item.value("content", "");

                if (entry.id.empty() || entry.content.empty()) {
                    LOG_ERROR("server.loading", "[Ollama Chat RAG] Entry missing required 'id' or 'content' field in file: {}", filePath);
                    continue;
                }

                // Load keywords array
                if (item.contains("keywords") && item["keywords"].is_array()) {
                    for (const auto& keyword : item["keywords"]) {
                        entry.keywords.push_back(keyword.get<std::string>());
                    }
                }

                // Load tags array
                if (item.contains("tags") && item["tags"].is_array()) {
                    for (const auto& tag : item["tags"]) {
                        entry.tags.push_back(tag.get<std::string>());
                    }
                }

                // Load references array (ids of related entries)
                if (item.contains("references") && item["references"].is_array()) {
                    for (const auto& ref : item["references"]) {
                        entry.references.push_back(ref.get<std::string>());
                    }
                }

                m_ragEntries.push_back(entry);
                entriesLoaded++;
            }
            catch (const std::exception& e) {
                LOG_ERROR("server.loading", "[Ollama Chat RAG] Error parsing entry in {}: {}", filePath, e.what());
            }
        }

        LOG_INFO("server.loading", "[Ollama Chat RAG] Loaded {} entries from {}", entriesLoaded, filePath);
        return entriesLoaded > 0;
    }
    catch (const std::exception& e) {
        LOG_ERROR("server.loading", "[Ollama Chat RAG] Error loading file {}: {}", filePath, e.what());
        return false;
    }
}

std::vector<RAGResult> OllamaRAGSystem::RetrieveRelevantInfo(const std::string& query, uint32_t maxResults, float similarityThreshold)
{
    std::vector<RAGResult> results;

    if (!m_initialized || query.empty()) {
        return results;
    }

    if (g_RAGImprovedScoring)
    {
        // Build the query's sparse, L2-normalized TF-IDF vector once.
        std::unordered_map<std::string, float> tf;
        for (const auto& t : NormalizeTokens(query))
            tf[t] += 1.0f;
        std::unordered_map<std::string, float> queryVec;
        float norm = 0.0f;
        for (const auto& kv : tf)
        {
            auto it = m_idf.find(kv.first);
            float w = kv.second * (it != m_idf.end() ? it->second : 0.0f);
            if (w != 0.0f)
            {
                queryVec[kv.first] = w;
                norm += w * w;
            }
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f)
            for (auto& kv : queryVec)
                kv.second /= norm;

        for (size_t i = 0; i < m_ragEntries.size(); ++i)
        {
            float similarity = CalculateSimilarityImproved(queryVec, i);
            if (similarity >= similarityThreshold)
                results.push_back({&m_ragEntries[i], similarity});
        }
    }
    else
    {
        for (const auto& entry : m_ragEntries)
        {
            float similarity = CalculateSimilarity(query, entry);
            if (similarity >= similarityThreshold)
                results.push_back({&entry, similarity});
        }
    }

    // Sort by similarity (highest first)
    std::sort(results.begin(), results.end(),
              [](const RAGResult& a, const RAGResult& b) {
                  return a.similarity > b.similarity;
              });

    // Limit results: strict top-N, or (default) weighted-by-similarity sampling over a top-K pool
    // so the surfaced snippets vary across calls without losing relevance.
    if (results.size() > maxResults) {
        if (!g_RAGRandomizeSelection) {
            results.resize(maxResults);  // strict top-N (rollback path; original behavior)
        } else {
            // Pool = top-K by similarity (results already sorted desc). Never smaller than maxResults.
            size_t cap = std::max<uint32_t>(g_RAGSelectionPoolSize, maxResults);
            size_t poolSize = std::min<size_t>(results.size(), cap);
            std::vector<RAGResult> pool(results.begin(), results.begin() + poolSize);

            // Draw maxResults without replacement, probability proportional to similarity.
            std::vector<RAGResult> chosen;
            chosen.reserve(maxResults);
            while (chosen.size() < maxResults && !pool.empty()) {
                float total = 0.0f;
                for (const auto& r : pool) {
                    total += r.similarity;
                }
                float roll = (total > 0.0f) ? frand(0.0f, total) : 0.0f;
                size_t pick = 0;
                float cum = 0.0f;
                for (size_t i = 0; i < pool.size(); ++i) {
                    cum += pool[i].similarity;
                    if (roll <= cum) {
                        pick = i;
                        break;
                    }
                }
                chosen.push_back(pool[pick]);
                pool.erase(pool.begin() + pick);
            }
            results = std::move(chosen);
        }
    }

    // Expand one hop along references. ON (default) = per-direct-hit random sampling; OFF = legacy global FCFS.
    if (g_RAGExpandReferences && g_RAGMaxReferences > 0) {
        std::unordered_set<std::string> present;
        for (const auto& r : results) {
            present.insert(r.entry->id);
        }

        std::vector<RAGResult> refResults;

        if (!g_RAGRandomizeReferences) {
            // ---- legacy global FCFS pool (unchanged behavior; instant revert) ----
            for (const auto& r : results) {
                if (refResults.size() >= static_cast<size_t>(g_RAGMaxReferences)) {
                    break;
                }
                for (const auto& refId : r.entry->references) {
                    if (refResults.size() >= static_cast<size_t>(g_RAGMaxReferences)) {
                        break;
                    }
                    if (present.count(refId)) {
                        continue;
                    }
                    auto it = m_idIndex.find(refId);
                    if (it == m_idIndex.end()) {
                        continue;
                    }
                    present.insert(refId);
                    RAGResult rr;
                    rr.entry = it->second;
                    rr.similarity = 0.0f;
                    rr.isReference = true;
                    refResults.push_back(rr);
                }
            }
        } else {
            // ---- per-direct-hit random sampling (variety, no crowding) ----
            for (const auto& r : results) {
                std::vector<const std::string*> cand;
                for (const auto& refId : r.entry->references) {
                    if (present.count(refId)) {
                        continue;
                    }
                    if (m_idIndex.find(refId) == m_idIndex.end()) {
                        continue;   // skip dangling
                    }
                    cand.push_back(&refId);
                }
                if (cand.empty()) {
                    continue;
                }

                uint32_t take = std::min<uint32_t>(g_RAGMaxReferences, static_cast<uint32_t>(cand.size()));
                for (uint32_t i = 0; i < take; ++i) {                          // partial Fisher-Yates
                    uint32_t j = i + urand(0, static_cast<uint32_t>(cand.size()) - 1 - i);
                    std::swap(cand[i], cand[j]);
                }
                for (uint32_t i = 0; i < take; ++i) {
                    const std::string& refId = *cand[i];
                    if (present.count(refId)) {
                        continue;                        // cross-hit dedup
                    }
                    present.insert(refId);
                    RAGResult rr;
                    rr.entry = m_idIndex[refId];
                    rr.similarity = 0.0f;
                    rr.isReference = true;
                    refResults.push_back(rr);
                }
            }
        }

        results.insert(results.end(), refResults.begin(), refResults.end());
    }

    return results;
}

std::string OllamaRAGSystem::GetFormattedRAGInfo(const std::vector<RAGResult>& results)
{
    if (results.empty()) {
        return "";
    }

    std::stringstream ss;
    bool hasContent = false;
    std::vector<const RAGEntry*> refs;

    // Direct hits: full content, one per line. References collected for a trailing line.
    for (const auto& result : results) {
        if (result.isReference) {
            refs.push_back(result.entry);
            continue;
        }
        if (hasContent) {
            ss << "\n";
        }
        ss << "- " << result.entry->title << ": " << result.entry->content;
        hasContent = true;
    }

    // References: a single "Related:" line, title + short_description (title only if missing).
    if (!refs.empty()) {
        if (hasContent) {
            ss << "\n";
        }
        ss << "Related: ";
        for (size_t i = 0; i < refs.size(); ++i) {
            ss << refs[i]->title;
            if (!refs[i]->short_description.empty()) {
                ss << ", " << refs[i]->short_description;
            }
            if (i + 1 < refs.size()) {
                ss << "; ";
            }
        }
    }

    return ss.str();
}

float OllamaRAGSystem::CalculateSimilarity(const std::string& query, const RAGEntry& entry)
{
    // Combine entry content with keywords for better matching
    std::string entryText = entry.title + " " + entry.content;
    for (const auto& keyword : entry.keywords) {
        entryText += " " + keyword;
    }

    // Simple TF-IDF like similarity using term frequency vectors
    auto queryVector = TextToTFVector(PreprocessText(query), m_vocabulary);
    auto entryVector = TextToTFVector(PreprocessText(entryText), m_vocabulary);

    return CalculateCosineSimilarity(queryVector, entryVector);
}

std::string OllamaRAGSystem::PreprocessText(const std::string& text) const
{
    std::string result = text;
    // Convert to lowercase
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);

    // Remove punctuation (simple approach)
    result.erase(std::remove_if(result.begin(), result.end(),
                                [](char c) { return std::ispunct(c); }), result.end());

    return result;
}

std::vector<std::string> OllamaRAGSystem::TokenizeText(const std::string& text) const
{
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    while (ss >> token) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

float OllamaRAGSystem::CalculateCosineSimilarity(const std::vector<float>& vec1, const std::vector<float>& vec2) const
{
    if (vec1.size() != vec2.size()) {
        return 0.0f;
    }

    float dotProduct = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (size_t i = 0; i < vec1.size(); ++i) {
        dotProduct += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);

    if (norm1 == 0.0f || norm2 == 0.0f) {
        return 0.0f;
    }

    return dotProduct / (norm1 * norm2);
}

std::vector<float> OllamaRAGSystem::TextToTFVector(const std::string& text, const std::vector<std::string>& vocabulary) const
{
    auto tokens = TokenizeText(text);
    std::unordered_map<std::string, int> termFreq;

    // Count term frequencies
    for (const auto& token : tokens) {
        termFreq[token]++;
    }

    // Create TF vector
    std::vector<float> vector(vocabulary.size(), 0.0f);
    for (size_t i = 0; i < vocabulary.size(); ++i) {
        auto it = termFreq.find(vocabulary[i]);
        if (it != termFreq.end()) {
            vector[i] = static_cast<float>(it->second);
        }
    }

    return vector;
}

bool OllamaRAGSystem::IsStopword(const std::string& token) const
{
    static const std::unordered_set<std::string> kStop = {
        "a","an","the","of","to","and","or","in","on","at","is","are","was","were",
        "be","been","being","it","its","this","that","these","those","for","with",
        "as","by","from","into","but","not","no","do","does","did","has","have","had",
        "you","your","we","our","they","them","their","he","she","his","her","him",
        "i","me","my","what","which","who","whom","when","where","why","how","there",
        "here","then","than","so","if","up","out","about","over","also","can","will",
        "would","should","could","just","very","too","more","most"
    };
    return kStop.count(token) > 0;
}

std::string OllamaRAGSystem::Stem(const std::string& token) const
{
    if (token.size() < 4)
        return token;

    auto endsWith = [&](const char* suf, size_t n) {
        return token.size() >= n && token.compare(token.size() - n, n, suf) == 0;
    };
    auto tryStrip = [&](size_t n, const std::string& add) -> std::string {
        if (token.size() - n + add.size() >= 3)
            return token.substr(0, token.size() - n) + add;
        return token;
    };

    if (endsWith("ies", 3)) return tryStrip(3, "y");   // berries -> berry
    if (endsWith("es", 2))  return tryStrip(2, "");     // blades  -> blade(s) handled; horses -> hors? guard keeps len>=3
    if (endsWith("ed", 2))  return tryStrip(2, "");     // raided  -> raid
    if (endsWith("ing", 3)) return tryStrip(3, "");     // raiding -> raid
    if (endsWith("er", 2))  return tryStrip(2, "");     // miner   -> min? -> guard
    if (endsWith("s", 1))   return tryStrip(1, "");     // raids   -> raid
    return token;
}

std::vector<std::string> OllamaRAGSystem::NormalizeTokens(const std::string& text) const
{
    std::vector<std::string> out;
    for (const auto& raw : TokenizeText(PreprocessText(text)))
    {
        if (raw.empty() || IsStopword(raw))
            continue;
        std::string s = Stem(raw);
        if (!s.empty())
            out.push_back(s);
    }
    return out;
}

void OllamaRAGSystem::BuildIdf()
{
    m_idf.clear();
    const float N = static_cast<float>(m_ragEntries.size());
    std::unordered_map<std::string, uint32_t> df;
    for (const auto& entry : m_ragEntries)
    {
        std::string text = entry.title + " " + entry.content;
        for (const auto& kw : entry.keywords)
            text += " " + kw;
        std::unordered_set<std::string> seen;
        for (const auto& t : NormalizeTokens(text))
            seen.insert(t);                 // count each term once per entry
        for (const auto& t : seen)
            df[t]++;
    }
    for (const auto& kv : df)
        m_idf[kv.first] = std::log((N + 1.0f) / (static_cast<float>(kv.second) + 1.0f)) + 1.0f;
}

void OllamaRAGSystem::BuildEntryVectors()
{
    m_entryVectors.assign(m_ragEntries.size(), {});
    for (size_t i = 0; i < m_ragEntries.size(); ++i)
    {
        const auto& entry = m_ragEntries[i];
        std::string text = entry.title + " " + entry.content;
        for (const auto& kw : entry.keywords)
            text += " " + kw;

        std::unordered_map<std::string, float> tf;
        for (const auto& t : NormalizeTokens(text))
            tf[t] += 1.0f;

        auto& vec = m_entryVectors[i];
        float norm = 0.0f;
        for (const auto& kv : tf)
        {
            auto it = m_idf.find(kv.first);
            float w = kv.second * (it != m_idf.end() ? it->second : 0.0f);
            if (w != 0.0f)
            {
                vec[kv.first] = w;
                norm += w * w;
            }
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f)
            for (auto& kv : vec)
                kv.second /= norm;
    }
}

float OllamaRAGSystem::CalculateSimilarityImproved(
    const std::unordered_map<std::string, float>& queryVec, size_t entryIndex) const
{
    if (entryIndex >= m_entryVectors.size())
        return 0.0f;
    const auto& entryVec = m_entryVectors[entryIndex];
    float dot = 0.0f;
    for (const auto& kv : queryVec)
    {
        auto it = entryVec.find(kv.first);
        if (it != entryVec.end())
            dot += kv.second * it->second;
    }
    return dot;   // both operands are pre-L2-normalized -> dot == cosine
}