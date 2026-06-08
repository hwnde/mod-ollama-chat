#ifndef MOD_OLLAMA_CHAT_QUERYMANAGER_H
#define MOD_OLLAMA_CHAT_QUERYMANAGER_H

#include <atomic>
#include <chrono>
#include <string>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

std::string QueryOllamaAPI(const std::string& prompt, bool applySalt);

class QueryManager {
public:
    QueryManager();
    void setMaxConcurrentQueries(int maxQueries);
    std::future<std::string> submitQuery(const std::string& prompt);

private:
    struct QueryTask {
        std::string prompt;
        std::promise<std::string> promise;
        std::chrono::steady_clock::time_point enqueuedAt;
    };

    void processQuery(const std::string& prompt, std::promise<std::string> promise);

    int maxConcurrentQueries; // 0 means no limit
    int currentQueries;
    std::mutex mutex_;
    std::queue<QueryTask> taskQueue;
    std::atomic<uint64_t> droppedFull{0};
    std::atomic<uint64_t> droppedStale{0};
};

#endif // MOD_OLLAMA_CHAT_QUERYMANAGER_H
