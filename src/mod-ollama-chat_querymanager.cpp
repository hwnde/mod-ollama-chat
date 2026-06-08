#include "mod-ollama-chat_querymanager.h"
#include "mod-ollama-chat_config.h"  // For g_MaxConcurrentQueries
#include "Log.h"
#include <thread>
#include <vector>

// Constructor: initialize with the configuration value.
QueryManager::QueryManager()
    : maxConcurrentQueries(g_MaxConcurrentQueries), currentQueries(0)
{
}

// Set maximum concurrent queries (0 means no limit).
void QueryManager::setMaxConcurrentQueries(int maxQueries) {
    std::lock_guard<std::mutex> lock(mutex_);
    maxConcurrentQueries = maxQueries;
}

// Submit a query and return a future for the result.
std::future<std::string> QueryManager::submitQuery(const std::string& prompt) {
    std::promise<std::string> promise;
    std::future<std::string> future = promise.get_future();

    bool shouldRunNow = false;
    bool rejectedFull = false;
    size_t depthSnapshot = 0;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (maxConcurrentQueries == 0 || currentQueries < maxConcurrentQueries) {
            ++currentQueries;
            shouldRunNow = true;
        } else if (g_OllamaQueueMaxDepth > 0 &&
                   taskQueue.size() >= static_cast<size_t>(g_OllamaQueueMaxDepth)) {
            rejectedFull = true;
            depthSnapshot = taskQueue.size();
        } else {
            taskQueue.push({ prompt, std::move(promise), std::chrono::steady_clock::now() });
        }
    }

    if (shouldRunNow) {
        std::thread(&QueryManager::processQuery, this, prompt, std::move(promise)).detach();
    } else if (rejectedFull) {
        promise.set_value("");
        ++droppedFull;
        if (g_DebugEnabled) {
            LOG_INFO("server.loading", "[Ollama Chat] Query dropped: full (depth={}/max={}, droppedFull={})",
                     depthSnapshot, g_OllamaQueueMaxDepth, droppedFull.load());
        }
    }

    return future;
}

// Process the query by calling the API and then handling any queued tasks.
void QueryManager::processQuery(const std::string& prompt, std::promise<std::string> promise) {
    std::string result = QueryOllamaAPI(prompt, true);
    promise.set_value(result);

    std::vector<std::promise<std::string>> stalePromises;  // fulfilled after the lock
    std::string nextPrompt;
    std::promise<std::string> nextPromise;
    bool haveNext = false;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        --currentQueries;
        // Pull the next runnable task, skipping any that have aged past the deadline.
        while (!taskQueue.empty() && (maxConcurrentQueries == 0 || currentQueries < maxConcurrentQueries)) {
            QueryTask task = std::move(taskQueue.front());
            taskQueue.pop();

            if (g_OllamaQueueMaxAgeMs > 0) {
                auto ageMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - task.enqueuedAt).count();
                if (ageMs > g_OllamaQueueMaxAgeMs) {
                    stalePromises.push_back(std::move(task.promise));  // drop: fulfil "" after lock
                    continue;  // slot still free — keep looking, do NOT ++currentQueries
                }
            }

            // Fresh task — claim the slot and dispatch it after the lock.
            ++currentQueries;
            nextPrompt = task.prompt;
            nextPromise = std::move(task.promise);
            haveNext = true;
            break;
        }
    }

    // Outside the lock: fulfil dropped promises, count/log, then dispatch the fresh one.
    if (!stalePromises.empty()) {
        for (auto& p : stalePromises)
            p.set_value("");
        droppedStale += stalePromises.size();
        if (g_DebugEnabled) {
            LOG_INFO("server.loading", "[Ollama Chat] Query dropped: stale ({} dropped, maxAgeMs={}, droppedStale={})",
                     (uint32)stalePromises.size(), g_OllamaQueueMaxAgeMs, droppedStale.load());
        }
    }
    if (haveNext) {
        std::thread(&QueryManager::processQuery, this, nextPrompt, std::move(nextPromise)).detach();
    }
}
