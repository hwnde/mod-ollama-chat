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

// Priority lane for a queued NPU query. High = player-directed replies (preempt);
// Normal = ambient chatter + sentiment. Only honored when OllamaChat.PrioritizeReplies = 1.
enum class QueryPriority { Normal, High };

class QueryManager {
public:
    QueryManager();
    void setMaxConcurrentQueries(int maxQueries);
    std::future<std::string> submitQuery(const std::string& prompt,
                                         QueryPriority prio = QueryPriority::Normal,
                                         bool applySalt = true);

private:
    struct QueryTask {
        std::string prompt;
        std::promise<std::string> promise;
        std::chrono::steady_clock::time_point enqueuedAt;
        bool applySalt;
    };

    void processQuery(const std::string& prompt, std::promise<std::string> promise, bool applySalt);

    int maxConcurrentQueries; // 0 means no limit
    int currentQueries;
    std::mutex mutex_;
    std::queue<QueryTask> highQueue;     // replies (drained first when prioritizing)
    std::queue<QueryTask> normalQueue;   // ambient chatter + sentiment
    std::atomic<uint64_t> droppedFull{0};
    std::atomic<uint64_t> droppedStale{0};
    std::atomic<uint64_t> droppedEvicted{0};
};

#endif // MOD_OLLAMA_CHAT_QUERYMANAGER_H
