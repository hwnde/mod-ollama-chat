#ifndef MOD_OLLAMA_CHAT_API_H
#define MOD_OLLAMA_CHAT_API_H

#include <string>
#include <future>
#include "mod-ollama-chat_querymanager.h"

std::string QueryOllamaAPI(const std::string& prompt, bool applySalt);

// Debug-only sanity check of the streaming soft-stop accumulator (no network).
void SoftStopSelfTest();

// Trims runaway NPU continuation past the real answer (fake turns, chat-template tokens, wiki artifacts).
// No-op when OllamaChat.TrimRunaway = 0 or RunawayPatterns is empty.
std::string TrimRunawayGeneration(const std::string& reply);

// Boot self-test for TrimRunawayGeneration.
void RunawayTrimSelfTest();

// Checks if an API response is valid (not an error message)
bool IsValidAPIResponse(const std::string& response);

// Submits a query to the API at the given priority (Normal = ambient/sentiment, High = replies).
// applySalt mirrors QueryOllamaAPI's param (chat = true, sentiment = false) and rides through the queue.
std::future<std::string> SubmitQuery(const std::string& prompt,
                                     QueryPriority prio = QueryPriority::Normal,
                                     bool applySalt = true);

// Declare the global QueryManager variable.
extern QueryManager g_queryManager;

#endif // MOD_OLLAMA_CHAT_API_H
