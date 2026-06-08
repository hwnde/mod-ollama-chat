#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_httpclient.h"
#include "mod-ollama-chat-utilities.h"
#include "Log.h"
#include <sstream>
#include <nlohmann/json.hpp>
#include <fmt/core.h>
#include <thread>
#include <mutex>
#include <queue>
#include <future>
#include <algorithm>
#include <random>

std::string ExtractTextBetweenDoubleQuotes(const std::string& response)
{
    size_t first = response.find('"');
    size_t second = response.find('"', first + 1);
    if (first != std::string::npos && second != std::string::npos) {
        return response.substr(first + 1, second - first - 1);
    }
    return response;
}

static const uint32_t SOFT_STOP_HEADROOM = 32;

static bool EndsOnSentenceBoundary(const std::string& s)
{
    size_t e = s.size();
    while (e > 0)
    {
        unsigned char c = (unsigned char)s[e - 1];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
            c == '"' || c == '\'' || c == ')' || c == ']' || c == '*')
        { --e; continue; }
        break;
    }
    if (e == 0)
        return false;
    char c = s[e - 1];
    if (c == '.' || c == '!' || c == '?')
        return true;
    if (e >= 3 &&
        (unsigned char)s[e - 3] == 0xE2 &&
        (unsigned char)s[e - 2] == 0x80 &&
        (unsigned char)s[e - 1] == 0xA6)
        return true;
    return false;
}

struct OllamaStreamAccumulator
{
    std::string lineBuf;
    std::string text;
    uint32_t    tokens = 0;
    uint32_t    softTarget = 0;
    bool        done = false;

    explicit OllamaStreamAccumulator(uint32_t target) : softTarget(target) {}

    bool ConsumeLine(const std::string& raw)
    {
        size_t b = raw.find_first_not_of(" \t\r\n");
        if (b == std::string::npos)
            return false;
        size_t e = raw.find_last_not_of(" \t\r\n");
        std::string line = raw.substr(b, e - b + 1);
        try
        {
            nlohmann::json j = nlohmann::json::parse(line);
            std::string piece;
            if (j.contains("message") && j["message"].is_object()
                && j["message"].contains("content") && j["message"]["content"].is_string())
                piece = j["message"]["content"].get<std::string>();   // /api/chat
            else if (j.contains("response") && j["response"].is_string())
                piece = j["response"].get<std::string>();             // /api/generate
            if (!piece.empty())
            {
                text += piece;
                ++tokens;
            }
            if (j.contains("done") && j["done"].is_boolean() && j["done"].get<bool>())
                done = true;
        }
        catch (const std::exception&)
        {
        }
        if (done)
            return true;
        if (softTarget > 0 && tokens >= softTarget && EndsOnSentenceBoundary(text))
            return true;
        return false;
    }

    bool Feed(const char* data, size_t len)
    {
        lineBuf.append(data, len);
        size_t nl;
        while ((nl = lineBuf.find('\n')) != std::string::npos)
        {
            std::string line = lineBuf.substr(0, nl);
            lineBuf.erase(0, nl + 1);
            if (ConsumeLine(line))
                return true;
        }
        return false;
    }
};

// Cut the reply at the earliest runaway/stop marker the NPU sometimes emits past
// the real answer (fake "User:"/"Assistant:" turns, chat-template tokens, wiki artifacts).
// Patterns are literal substrings from OllamaChat.RunawayPatterns. The config stores
// the two-character sequence backslash-n for newline markers; expand those here so a
// real '\n' in the model text matches the configured "\n\nUser:" style pattern.
static std::string ExpandEscapes(const std::string& in)
{
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        if (in[i] == '\\' && i + 1 < in.size() && in[i + 1] == 'n')
        {
            out.push_back('\n');
            ++i;
        }
        else
            out.push_back(in[i]);
    }
    return out;
}

std::string TrimRunawayGeneration(const std::string& reply)
{
    if (!g_TrimRunaway || g_RunawayPatterns.empty() || reply.empty())
        return reply;

    size_t cut = std::string::npos;
    for (const std::string& rawPat : g_RunawayPatterns)
    {
        std::string pat = ExpandEscapes(rawPat);
        if (pat.empty())
            continue;
        size_t pos = reply.find(pat);
        if (pos != std::string::npos && pos < cut)
            cut = pos;
    }
    if (cut == std::string::npos)
        return reply;

    std::string trimmed = reply.substr(0, cut);
    size_t end = trimmed.find_last_not_of(" \t\r\n");
    return (end == std::string::npos) ? std::string() : trimmed.substr(0, end + 1);
}

void SoftStopSelfTest()
{
    int passed = 0, total = 0;

    auto feedAll = [](OllamaStreamAccumulator& acc, const std::string& stream) -> bool {
        return acc.Feed(stream.data(), stream.size());
    };

    {
        ++total;
        OllamaStreamAccumulator acc(2);
        std::string s =
            "{\"response\":\"Hello\"}\n{\"response\":\" world\"}\n{\"response\":\".\"}\n{\"response\":\" extra\"}\n";
        bool stopped = feedAll(acc, s);
        if (stopped && acc.text == "Hello world.") ++passed;
        else LOG_ERROR("server.loading", "[Ollama Chat] SoftStop self-test FAIL (stop-at-period) -> stopped={} text=[{}]", stopped, acc.text);
    }

    {
        ++total;
        OllamaStreamAccumulator acc(2);
        std::string s =
            "{\"response\":\"Hello\"}\n{\"response\":\" world\"}\n{\"response\":\" today\"}\n{\"response\":\"!\"}\n";
        bool stopped = feedAll(acc, s);
        if (stopped && acc.text == "Hello world today!") ++passed;
        else LOG_ERROR("server.loading", "[Ollama Chat] SoftStop self-test FAIL (wait-for-boundary) -> stopped={} text=[{}]", stopped, acc.text);
    }

    {
        ++total;
        OllamaStreamAccumulator acc(1);
        std::string p1 = "{\"respo";
        std::string p2 = "nse\":\"Hi.\"}\n";
        bool s1 = acc.Feed(p1.data(), p1.size());
        bool s2 = acc.Feed(p2.data(), p2.size());
        if (!s1 && s2 && acc.text == "Hi.") ++passed;
        else LOG_ERROR("server.loading", "[Ollama Chat] SoftStop self-test FAIL (chunk-split) -> s1={} s2={} text=[{}]", s1, s2, acc.text);
    }

    {
        ++total;
        OllamaStreamAccumulator acc(10);
        std::string s = "{\"response\":\"Hi\",\"done\":false}\n{\"response\":\" bye\",\"done\":true}\n";
        bool stopped = feedAll(acc, s);
        if (stopped && acc.text == "Hi bye") ++passed;
        else LOG_ERROR("server.loading", "[Ollama Chat] SoftStop self-test FAIL (done-flag) -> stopped={} text=[{}]", stopped, acc.text);
    }

    {   // chat shape: stop at sentence boundary
        ++total;
        OllamaStreamAccumulator acc(2);
        std::string s =
            "{\"message\":{\"content\":\"Hello\"}}\n{\"message\":{\"content\":\" world\"}}\n{\"message\":{\"content\":\".\"}}\n{\"message\":{\"content\":\" extra\"}}\n";
        bool stopped = feedAll(acc, s);
        if (stopped && acc.text == "Hello world.") ++passed;
        else LOG_ERROR("server.loading", "[Ollama Chat] SoftStop self-test FAIL (chat stop-at-period) -> stopped={} text=[{}]", stopped, acc.text);
    }

    {   // chat shape: done flag terminates
        ++total;
        OllamaStreamAccumulator acc(10);
        std::string s = "{\"message\":{\"content\":\"Hi\"},\"done\":false}\n{\"message\":{\"content\":\" bye\"},\"done\":true}\n";
        bool stopped = feedAll(acc, s);
        if (stopped && acc.text == "Hi bye") ++passed;
        else LOG_ERROR("server.loading", "[Ollama Chat] SoftStop self-test FAIL (chat done-flag) -> stopped={} text=[{}]", stopped, acc.text);
    }

    LOG_INFO("server.loading", "[Ollama Chat] SoftStop self-test: {}/{} passed", passed, total);
}

void RunawayTrimSelfTest()
{
    int passed = 0, total = 0;
    // Save/restore the live config so the test is hermetic.
    bool savedEnable = g_TrimRunaway;
    std::vector<std::string> savedPatterns = g_RunawayPatterns;
    g_TrimRunaway = true;
    g_RunawayPatterns = { "\\n\\nUser:", "<|eot_id|>", "\\n[[", "Wikipedia" };

    auto check = [&](const std::string& in, const std::string& want, const char* name) {
        ++total;
        std::string got = TrimRunawayGeneration(in);
        if (got == want) ++passed;
        else LOG_ERROR("server.loading", "[Ollama Chat] RunawayTrim self-test FAIL ({}) -> got=[{}] want=[{}]", name, got, want);
    };

    check("A clean line.", "A clean line.", "no-marker");
    check("Hello there.\n\nUser: ignore this", "Hello there.", "fake-user-turn");
    check("Bye.<|eot_id|>junk", "Bye.", "eot-token");
    check("Real reply.\n[[Special:Foo]]", "Real reply.", "wiki-link");

    LOG_INFO("server.loading", "[Ollama Chat] RunawayTrim self-test: {}/{} passed", passed, total);

    g_TrimRunaway = savedEnable;
    g_RunawayPatterns = savedPatterns;
}

// Per-request salt: the Lemonade hybrid-NPU backend decodes deterministically, so two
// bots fed an identical prompt say an identical line. A unique opaque tag makes each
// chat request a distinct input -> a distinct deterministic answer. thread_local RNG:
// QueryOllamaAPI runs on detached worker threads (no shared-state contention).
static std::string GenerateSalt()
{
    static thread_local std::mt19937 rng(std::random_device{}());
    return fmt::format("req-{:08x}", static_cast<uint32_t>(rng()));
}

// Function to perform the API call.
std::string QueryOllamaAPI(const std::string& prompt, bool applySalt)
{
    // Initialize our custom HTTP client
    static OllamaHttpClient httpClient;
    
    if (!httpClient.IsAvailable())
    {
        LOG_ERROR("server.loading", "[OllamaChat] ERROR: HTTP client not available. Check if Ollama service is running and accessible.");
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[OllamaChat] Debug: HTTP client initialization failed.");
        }
        return "";
    }

    std::string url   = g_OllamaUrl;
    std::string model = g_OllamaModel;

    // Sanitize the prompt to ensure it's valid UTF-8 before creating JSON
    std::string sanitizedPrompt = SanitizeUTF8(prompt);

    // Salt appended after any trailing /no_think — validated to leave /no_think effective.
    // Only when a system prompt exists (chat mode requires one; empty already WARNs at boot).
    std::string sysPrompt = g_OllamaSystemPrompt;
    if (g_PromptSalt && applySalt && !sysPrompt.empty())
    {
        std::string tag = GenerateSalt();
        sysPrompt += " [" + tag + "]";
        if (g_DebugEnabled)
            LOG_INFO("server.loading", "[Ollama Chat] Prompt salt tag: {}", tag);
    }

    nlohmann::json requestData;
    requestData["model"]  = model;
    requestData["stream"] = false;
    if (g_ApiMode == API_CHAT)
    {
        nlohmann::json messages = nlohmann::json::array();
        if (!sysPrompt.empty())
            messages.push_back({{"role", "system"}, {"content", SanitizeUTF8(sysPrompt)}});
        messages.push_back({{"role", "user"}, {"content", sanitizedPrompt}});
        requestData["messages"] = messages;
    }
    else
    {
        requestData["prompt"] = sanitizedPrompt;   // /api/generate -- unchanged
    }

    // Create options object for model parameters
    nlohmann::json options;
    bool hasOptions = false;

    // Only include if set (do not send defaults if user did not set them)
    if (g_OllamaNumPredict > 0) {
        options["num_predict"] = g_OllamaNumPredict;
        hasOptions = true;
    }
    if (g_OllamaTemperature != 0.8f) {
        options["temperature"] = g_OllamaTemperature;
        hasOptions = true;
    }
    if (g_OllamaTopP != 0.95f) {
        options["top_p"] = g_OllamaTopP;
        hasOptions = true;
    }
    if (g_OllamaRepeatPenalty != 1.1f) {
        options["repeat_penalty"] = g_OllamaRepeatPenalty;
        hasOptions = true;
    }
    if (g_OllamaTopK > 0) {
        options["top_k"] = g_OllamaTopK;
        hasOptions = true;
    }
    if (g_OllamaMinP != 0.0f) {
        options["min_p"] = g_OllamaMinP;
        hasOptions = true;
    }
    if (g_OllamaPresencePenalty != 0.0f) {
        options["presence_penalty"] = g_OllamaPresencePenalty;
        hasOptions = true;
    }
    if (g_OllamaFrequencyPenalty != 0.0f) {
        options["frequency_penalty"] = g_OllamaFrequencyPenalty;
        hasOptions = true;
    }
    if (g_OllamaNumCtx > 0) {
        options["num_ctx"] = g_OllamaNumCtx;
        hasOptions = true;
    }
    if (g_OllamaNumThreads > 0) {
        options["num_thread"] = g_OllamaNumThreads;
        hasOptions = true;
        if(g_DebugEnabled) {
            //LOG_INFO("server.loading", "[Ollama Chat] Setting num_thread to: {}", g_OllamaNumThreads);
        }
    } else if(g_DebugEnabled) {
        //LOG_INFO("server.loading", "[Ollama Chat] g_OllamaNumThreads is: {} (not sending num_thread)", g_OllamaNumThreads);
    }
    if (!g_OllamaSeed.empty()) {
        try {
            int seedValue = std::stoi(g_OllamaSeed);
            options["seed"] = seedValue; 
            hasOptions = true;
        } catch (const std::exception& e) {
            if(g_DebugEnabled) {
                LOG_INFO("server.loading", "[Ollama Chat] Invalid seed value: {}", g_OllamaSeed);
            }
        }
    }

    // Add options object if any options were set
    if (hasOptions) {
        requestData["options"] = options;
    }

    // Root-level parameters (these stay at root level)
    if (!g_OllamaStop.empty()) {
        // If comma-separated, convert to array
        std::vector<std::string> stopSeqs;
        std::stringstream ss(g_OllamaStop);
        std::string item;
        while (std::getline(ss, item, ',')) {
            // trim whitespace
            size_t start = item.find_first_not_of(" \t");
            size_t end = item.find_last_not_of(" \t");
            if (start != std::string::npos && end != std::string::npos)
                stopSeqs.push_back(item.substr(start, end - start + 1));
        }
        if (!stopSeqs.empty())
        {
            if (g_ApiMode == API_CHAT)
                requestData["options"]["stop"] = stopSeqs;   // chat: stop lives under options
            else
                requestData["stop"] = stopSeqs;              // generate: unchanged (root)
        }
    }
    if (g_ApiMode == API_GENERATE && !sysPrompt.empty())
    {
        // Sanitize system prompt as well
        requestData["system"] = SanitizeUTF8(sysPrompt);
    }

    if (g_ThinkModeEnableForModule)
    {
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] LLM set to Think mode.");
        }
        requestData["think"] = true;
        requestData["hidethinking"] = true;
    }

    std::string botReply;

    bool useStreaming = (g_SoftStopEnable && g_OllamaNumPredict > 0 && !g_ThinkModeEnableForModule);

    if (useStreaming)
    {
        requestData["stream"] = true;
        requestData["options"]["num_predict"] = g_OllamaNumPredict + SOFT_STOP_HEADROOM;

        std::string requestDataStr = requestData.dump();

        OllamaStreamAccumulator acc(g_OllamaNumPredict);
        bool ok = httpClient.PostStreaming(url, requestDataStr,
            [&](const char* data, size_t len) -> bool {
                return !acc.Feed(data, len);
            });

        if (!ok && acc.text.empty())
        {
            LOG_ERROR("server.loading", "[OllamaChat] ERROR: Streaming request to {} produced no output.", url);
            return "";
        }
        botReply = acc.text;

        if (g_DebugEnabled)
            LOG_INFO("server.loading", "[Ollama Chat] SoftStop streamed {} tokens (soft target {}).", acc.tokens, g_OllamaNumPredict);
    }
    else
    {
        std::string requestDataStr = requestData.dump();

        // Make HTTP POST request using our custom client
        std::string responseBuffer = httpClient.Post(url, requestDataStr);

        if (responseBuffer.empty())
        {
            LOG_ERROR("server.loading", "[OllamaChat] ERROR: Failed to reach Ollama API at {}. Check URL configuration and network connectivity.", url);
            if(g_DebugEnabled)
            {
                LOG_INFO("server.loading", "[OllamaChat] Debug: Empty response buffer from HTTP client. Model: {}", model);
            }
            return "";
        }

        std::stringstream ss(responseBuffer);
        std::string line;
        std::ostringstream extractedResponse;

        try
        {
            while (std::getline(ss, line))
            {
                if (line.empty() || std::all_of(line.begin(), line.end(), isspace))
                    continue;

                nlohmann::json jsonResponse = nlohmann::json::parse(line);

                if (g_ApiMode == API_CHAT)
                {
                    if (jsonResponse.contains("message") && jsonResponse["message"].is_object()
                        && jsonResponse["message"].contains("content") && jsonResponse["message"]["content"].is_string())
                        extractedResponse << jsonResponse["message"]["content"].get<std::string>();
                }
                else if (jsonResponse.contains("response") && jsonResponse["response"].is_string() && !jsonResponse["response"].get<std::string>().empty())
                {
                    extractedResponse << jsonResponse["response"].get<std::string>();
                }
            }
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("server.loading", "[OllamaChat] ERROR: JSON parsing failed. Exception: {}", e.what());
            if(g_DebugEnabled)
            {
                LOG_INFO("server.loading", "[OllamaChat] Debug: Response buffer content: {}", responseBuffer);
            }
            return "";
        }
        botReply = extractedResponse.str();
    }

    botReply = ExtractTextBetweenDoubleQuotes(botReply);
    botReply = TrimRunawayGeneration(botReply);

    // Check for unclosed think tags
    if (botReply.find("<think>") != std::string::npos || botReply.find("</think>") != std::string::npos)
    {
        LOG_ERROR("server.loading", "[OllamaChat] ERROR: Unclosed <think> tags detected in response. This usually means the model's output was truncated.");
        LOG_ERROR("server.loading", "[OllamaChat] SOLUTION: Set 'OllamaChat.ThinkModeEnableForModule = 1' in mod_ollama_chat.conf");
        LOG_ERROR("server.loading", "[OllamaChat] SOLUTION: Set 'OllamaChat.NumPredict = 0' (unlimited tokens) in mod_ollama_chat.conf");
        LOG_ERROR("server.loading", "[OllamaChat] SOLUTION: Set 'OllamaChat.NumCtx = 0' (model default context) in mod_ollama_chat.conf");
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[OllamaChat] Debug: Partial response with think tags: {}", botReply);
        }
        return "";
    }

    if (botReply.empty())
    {
        LOG_ERROR("server.loading", "[OllamaChat] ERROR: Empty response extracted from API. Model may not have generated any output.");
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[OllamaChat] Debug: Raw extracted response was empty.");
        }
        return "";
    }

    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[Ollama Chat] Parsed bot response: {}", botReply);

        if (g_ThinkModeEnableForModule)
        {
            if(g_DebugEnabled)
            {
                LOG_INFO("server.loading", "[Ollama Chat] Bot used think.");
            }
        }
    }

    return botReply;
}

// Helper function to check if a response is valid (not empty and not an error)
bool IsValidAPIResponse(const std::string& response)
{
    if (response.empty())
    {
        return false;
    }
    // Response is valid if it's not empty
    return true;
}

QueryManager g_queryManager;

// Interface function to submit a query.
std::future<std::string> SubmitQuery(const std::string& prompt)
{
    return g_queryManager.submitQuery(prompt);
}