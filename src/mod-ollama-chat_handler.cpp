#include "Log.h"
#include "Language.h"
#include "Player.h"
#include "Chat.h"
#include "Channel.h"
#include "PlayerbotAI.h"
#include "PlayerbotMgr.h"
#include "Config.h"
#include "Common.h"
#include "Guild.h"
#include "GuildMgr.h"
#include "ObjectAccessor.h"
#include "World.h"
#include "AiFactory.h"
#include "ChannelMgr.h"
#include <sstream>
#include <vector>
#include <unordered_set>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <thread>
#include <algorithm>
#include <random>
#include <cctype>
#include <chrono>
#include <ctime>
#include "DatabaseEnv.h"
#include "mod-ollama-chat_handler.h"
#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat_personality.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat-utilities.h"
#include "mod-ollama-chat_sentiment.h"
#include "mod-ollama-chat_rag.h"
#include <iomanip>
#include "SpellMgr.h"
#include "SpellInfo.h"
#include "SharedDefines.h"
#include "Group.h"
#include "Creature.h"
#include "GameObject.h"
#include "TravelMgr.h"
#include "TravelNode.h"
#include "ObjectMgr.h"
#include "QuestDef.h"

// For AzerothCore range checks
#include "GridNotifiersImpl.h"
#include "CellImpl.h"
#include "Map.h"
#include "GridNotifiers.h"

// Forward declarations for internal helper functions.
static bool IsBotEligibleForChatChannelLocal(Player* bot, Player* player,
                                             ChatChannelSourceLocal source, Channel* channel = nullptr, Player* receiver = nullptr);
static std::string GenerateBotPrompt(Player* bot, std::string playerMessage, Player* player, ChannelCategory channelCat);

// Helper function to format class name for any player
static std::string FormatPlayerClass(uint8_t classId)
{
    switch (classId)
    {
        case CLASS_WARRIOR:      return "Warrior";
        case CLASS_PALADIN:      return "Paladin";
        case CLASS_HUNTER:       return "Hunter";
        case CLASS_ROGUE:        return "Rogue";
        case CLASS_PRIEST:       return "Priest";
        case CLASS_DEATH_KNIGHT: return "Death Knight";
        case CLASS_SHAMAN:       return "Shaman";
        case CLASS_MAGE:         return "Mage";
        case CLASS_WARLOCK:      return "Warlock";
        case CLASS_DRUID:        return "Druid";
        default:                 return "Unknown";
    }
}

// Helper function to format race name for any player
static std::string FormatPlayerRace(uint8_t raceId)
{
    switch (raceId)
    {
        case RACE_HUMAN:         return "Human";
        case RACE_ORC:           return "Orc";
        case RACE_DWARF:         return "Dwarf";
        case RACE_NIGHTELF:      return "Night Elf";
        case RACE_UNDEAD_PLAYER: return "Undead";
        case RACE_TAUREN:        return "Tauren";
        case RACE_GNOME:         return "Gnome";
        case RACE_TROLL:         return "Troll";
        case RACE_BLOODELF:      return "Blood Elf";
        case RACE_DRAENEI:       return "Draenei";
        default:                 return "Unknown";
    }
}

const char* ChatChannelSourceLocalStr[] =
{
    "Undefined",  // 0
    "Say",        // 1
    "Party",      // 2
    "Raid",       // 3
    "Guild",      // 4
    "Officer",    // 5
    "Yell",       // 6
    "Whisper",    // 7
    "Unknown8",   // 8
    "Unknown9",   // 9
    "Unknown10",  // 10
    "Unknown11",  // 11
    "Unknown12",  // 12
    "Unknown13",  // 13
    "Unknown14",  // 14
    "Unknown15",  // 15
    "Unknown16",  // 16
    "General"     // 17
};

std::string GetConversationEntryKey(uint64_t botGuid, uint64_t playerGuid, const std::string& playerMessage, const std::string& botReply)
{
    // Use a combination that guarantees uniqueness
    return SafeFormat("{}:{}:{}:{}", botGuid, playerGuid, playerMessage, botReply);
}

std::string rtrim(const std::string& s)
{
    const std::string whitespace = " \t\n\r,.!?;:";
    size_t end = s.find_last_not_of(whitespace);
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

ChatChannelSourceLocal GetChannelSourceLocal(uint32_t type)
{
    switch (type)
    {
        case CHAT_MSG_SAY:
            return SRC_SAY_LOCAL;
        case CHAT_MSG_PARTY:
        case CHAT_MSG_PARTY_LEADER:
            return SRC_PARTY_LOCAL;
        case CHAT_MSG_RAID:
        case CHAT_MSG_RAID_LEADER:
        case CHAT_MSG_RAID_WARNING:
            return SRC_RAID_LOCAL;
        case CHAT_MSG_GUILD:
            return SRC_GUILD_LOCAL;
        case CHAT_MSG_OFFICER:
            return SRC_OFFICER_LOCAL;
        case CHAT_MSG_YELL:
            return SRC_YELL_LOCAL;
        case CHAT_MSG_WHISPER:
        case CHAT_MSG_WHISPER_FOREIGN:
        case CHAT_MSG_WHISPER_INFORM:
            return SRC_WHISPER_LOCAL;
        case CHAT_MSG_CHANNEL:
            return SRC_GENERAL_LOCAL;
        default:
            return SRC_UNDEFINED_LOCAL;
    }
}

Channel* GetValidChannel(uint32_t teamId, const std::string& channelName, Player* player)
{
    ChannelMgr* cMgr = ChannelMgr::forTeam(static_cast<TeamId>(teamId));
    Channel* channel = cMgr->GetChannel(channelName, player);
    if (!channel)
    {
        if(g_DebugEnabled)
        {
            LOG_ERROR("server.loading", "[Ollama Chat] Channel '{}' not found for team {}", channelName, teamId);
        }
    }
    return channel;
}

bool PlayerBotChatHandler::OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg)
{
    if (!g_Enable)
        return true;

    ChatChannelSourceLocal sourceLocal = GetChannelSourceLocal(type);
    ProcessChat(player, type, lang, msg, sourceLocal, nullptr, nullptr);
    return true;
}

bool PlayerBotChatHandler::OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Group* /*group*/)
{
    if (!g_Enable)
        return true;

    ChatChannelSourceLocal sourceLocal = GetChannelSourceLocal(type);
    ProcessChat(player, type, lang, msg, sourceLocal, nullptr, nullptr);
    return true;
}

bool PlayerBotChatHandler::OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Guild* /*guild*/)
{
    if (!g_Enable)
        return true;

    ChatChannelSourceLocal sourceLocal = GetChannelSourceLocal(type);
    ProcessChat(player, type, lang, msg, sourceLocal, nullptr, nullptr);
    return true;
}

bool PlayerBotChatHandler::OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Channel* channel)
{
    if (!g_Enable)
        return true;

    ChatChannelSourceLocal sourceLocal = GetChannelSourceLocal(type);
    ProcessChat(player, type, lang, msg, sourceLocal, channel, nullptr);
    return true;
}

bool PlayerBotChatHandler::OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Player* receiver)
{
    // Only process if our module is enabled
    if (!g_Enable)
        return true;

    if (type == CHAT_MSG_WHISPER)
    {
        // Check if this is a valid whisper to a bot
        if (!receiver || !player || player == receiver)
            return true;

        // Check if sender is a bot - if so, don't trigger Ollama responses for bot-to-bot whispers
        PlayerbotAI* senderAI = PlayerbotsMgr::instance().GetPlayerbotAI(player);
        if (senderAI && senderAI->IsBotAI())
        {
            return true;
        }

        PlayerbotAI* receiverAI = PlayerbotsMgr::instance().GetPlayerbotAI(receiver);
        if (!receiverAI || !receiverAI->IsBotAI())
            return true;
    }

    if (g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[Ollama Chat] OnPlayerCanUseChat called: player={}, type={}, receiver={}",
            player->GetName(), type, receiver ? receiver->GetName() : "null");
    }

    // Process the chat immediately in OnPlayerCanUseChat to prevent double processing
    ChatChannelSourceLocal sourceLocal = GetChannelSourceLocal(type);
    ProcessChat(player, type, lang, msg, sourceLocal, nullptr, receiver);

    // Return false to prevent the message from being processed again in OnPlayerChat
    return true;
}

void AppendBotConversation(uint64_t botGuid, uint64_t playerGuid, const std::string& playerMessage, const std::string& botReply)
{
    std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
    auto& playerHistory = g_BotConversationHistory[botGuid][playerGuid];
    playerHistory.push_back({ playerMessage, botReply });
    while (playerHistory.size() > g_MaxConversationHistory)
    {
        playerHistory.pop_front();
    }

}

void AppendBotRecentReply(uint64_t botGuid, const std::string& reply)
{
    if (reply.empty())
        return;
    std::lock_guard<std::mutex> lock(g_RecentRepliesMutex);
    auto& recents = g_BotRecentReplies[botGuid];
    recents.push_back(reply);
    while (recents.size() > g_AntiRepetitionWindow)
        recents.pop_front();
}

std::string GetAntiRepetitionPrompt(uint64_t botGuid)
{
    if (!g_EnableAntiRepetition)
        return "";
    std::lock_guard<std::mutex> lock(g_RecentRepliesMutex);
    auto it = g_BotRecentReplies.find(botGuid);
    if (it == g_BotRecentReplies.end() || it->second.empty())
        return "";
    std::string list;
    bool first = true;
    for (const auto& r : it->second)
    {
        if (!first)
            list += " | ";
        list += "\"" + r + "\"";
        first = false;
    }
    return SafeFormat(g_AntiRepetitionTemplate, fmt::arg("recent_replies", list));
}

// Build a prompt fragment listing recent lines spoken by OTHER bots within Say range of the
// speaker, so co-located bots can be told not to echo each other. Reads the existing per-bot
// buffer only (no new recording). Critical section is a pure copy-out; formatting is done
// outside the lock.
std::string GetNearbyBotsRecentReplies(Player* speaker)
{
    if (!g_EnableCrossBotAntiRepetition || !speaker || !speaker->IsInWorld())
        return "";

    // Snapshot the speaker's own recent lines so we don't re-list what the per-bot prompt covers.
    std::unordered_set<std::string> ownLines;
    {
        std::lock_guard<std::mutex> lock(g_RecentRepliesMutex);
        auto it = g_BotRecentReplies.find(speaker->GetGUID().GetRawValue());
        if (it != g_BotRecentReplies.end())
            for (const auto& r : it->second)
                ownLines.insert(r);
    }

    // Find nearby bots (same proximity test as random-chatter Say gating).
    std::vector<Player*> nearbyBots;
    for (auto const& pair : ObjectAccessor::GetPlayers())
    {
        Player* p = pair.second;
        if (!p || !p->IsInWorld() || p == speaker)
            continue;
        if (p->GetMap() != speaker->GetMap())
            continue;
        if (!PlayerbotsMgr::instance().GetPlayerbotAI(p))
            continue;
        if (speaker->GetDistance(p) <= g_SayDistance)
            nearbyBots.push_back(p);
    }
    if (nearbyBots.empty())
        return "";

    // Copy out each nearby bot's most-recent Window lines under the lock; format later.
    std::vector<std::string> collected;
    {
        std::lock_guard<std::mutex> lock(g_RecentRepliesMutex);
        for (Player* b : nearbyBots)
        {
            auto it = g_BotRecentReplies.find(b->GetGUID().GetRawValue());
            if (it == g_BotRecentReplies.end() || it->second.empty())
                continue;
            const auto& dq = it->second;
            uint32_t n = 0;
            for (auto rit = dq.rbegin(); rit != dq.rend() && n < g_CrossBotAntiRepetitionWindow; ++rit, ++n)
                collected.push_back(*rit);
        }
    }

    // Dedup verbatim, drop anything the speaker itself recently said, cap total.
    std::vector<std::string> finalLines;
    std::unordered_set<std::string> seen;
    for (const auto& line : collected)
    {
        if (line.empty() || ownLines.count(line) || seen.count(line))
            continue;
        seen.insert(line);
        finalLines.push_back(line);
        if (finalLines.size() >= static_cast<size_t>(g_CrossBotAntiRepetitionMaxLines))
            break;
    }
    if (finalLines.empty())
        return "";

    std::string list;
    bool first = true;
    for (const auto& r : finalLines)
    {
        if (!first)
            list += " | ";
        list += "\"" + r + "\"";
        first = false;
    }
    return SafeFormat(g_CrossBotAntiRepetitionTemplate, fmt::arg("nearby_replies", list));
}

// Token-set Jaccard similarity of two lines (whitespace tokens). Used only by the CROSSREP
// measurement detector below.
static double CrossRepTokenJaccard(const std::string& a, const std::string& b)
{
    std::unordered_set<std::string> ta, tb;
    {
        std::stringstream ss(a);
        std::string t;
        while (ss >> t) ta.insert(t);
    }
    {
        std::stringstream ss(b);
        std::string t;
        while (ss >> t) tb.insert(t);
    }
    if (ta.empty() || tb.empty())
        return 0.0;
    size_t inter = 0;
    for (const auto& t : ta)
        if (tb.count(t))
            ++inter;
    size_t uni = ta.size() + tb.size() - inter;
    return uni ? static_cast<double>(inter) / static_cast<double>(uni) : 0.0;
}

// Measurement detector (DebugEnabled-gated, independent of the Enable toggle): logs when the
// speaker's just-emitted line verbatim- or high-overlap-matches a nearby bot's last line.
void LogCrossBotRepetition(Player* speaker, const std::string& emittedLine)
{
    if (!g_DebugEnabled || !speaker || !speaker->IsInWorld() || emittedLine.empty())
        return;
    for (auto const& pair : ObjectAccessor::GetPlayers())
    {
        Player* p = pair.second;
        if (!p || !p->IsInWorld() || p == speaker)
            continue;
        if (p->GetMap() != speaker->GetMap())
            continue;
        if (!PlayerbotsMgr::instance().GetPlayerbotAI(p))
            continue;
        if (speaker->GetDistance(p) > g_SayDistance)
            continue;
        std::string otherLast;
        {
            std::lock_guard<std::mutex> lock(g_RecentRepliesMutex);
            auto it = g_BotRecentReplies.find(p->GetGUID().GetRawValue());
            if (it != g_BotRecentReplies.end() && !it->second.empty())
                otherLast = it->second.back();
        }
        if (otherLast.empty())
            continue;
        if (otherLast == emittedLine || CrossRepTokenJaccard(emittedLine, otherLast) >= 0.8)
            LOG_INFO("server.loading",
                "[Ollama Chat] CROSSREP hit: {} echoed {} within {}y :: \"{}\"",
                speaker->GetName(), p->GetName(), speaker->GetDistance(p), emittedLine);
    }
}

std::string ComputeChannelKey(ChatChannelSourceLocal sourceLocal, Channel* channel, Player* sender)
{
    if (!sender)
        return "";
    switch (sourceLocal)
    {
        case SRC_PARTY_LOCAL:
        case SRC_RAID_LOCAL:
            if (Group* g = sender->GetGroup())
                return "P:" + std::to_string(g->GetGUID().GetRawValue());
            return "";
        case SRC_GUILD_LOCAL:
        case SRC_OFFICER_LOCAL:
            if (uint32 gid = sender->GetGuildId())
                return "G:" + std::to_string(gid);
            return "";
        case SRC_GENERAL_LOCAL:
            if (channel)
                return "C:" + channel->GetName();
            return "";
        case SRC_SAY_LOCAL:
        case SRC_YELL_LOCAL:
            // Proximity chat has no Channel*; scope a shared buffer to the sender's
            // map+zone. Say and Yell share one "recent talk around here" thread, and the
            // team-id prefix keeps it single-faction (matches eligibility filtering).
            // Zone is coarser than Say's true ~25yd range, but an acceptable v1 approximation.
            return "S:" + std::to_string(sender->GetTeamId())
                 + ":" + std::to_string(sender->GetMapId())
                 + ":" + std::to_string(sender->GetZoneId());
        default:
            return "";
    }
}

ChannelCategory ClassifyChannel(ChatChannelSourceLocal sourceLocal, Channel* channel)
{
    switch (sourceLocal)
    {
        case SRC_GUILD_LOCAL:
        case SRC_OFFICER_LOCAL:
            return ChannelCategory::Guild;
        case SRC_PARTY_LOCAL:
            return ChannelCategory::Party;
        case SRC_RAID_LOCAL:
            return ChannelCategory::Raid;
        case SRC_SAY_LOCAL:
            return ChannelCategory::Say;
        case SRC_YELL_LOCAL:
            return ChannelCategory::Yell;
        case SRC_GENERAL_LOCAL:
            if (channel)
            {
                const std::string& name = channel->GetName();
                if (name.find("Trade") != std::string::npos)
                    return ChannelCategory::Trade;
                if (name.find("General") != std::string::npos)
                    return ChannelCategory::General;
            }
            return ChannelCategory::Others;
        default:
            return ChannelCategory::Others;
    }
}

std::string GetChannelFrame(ChannelCategory cat)
{
    size_t idx = static_cast<size_t>(cat);
    if (idx < 8 && !g_ChannelFrames[idx].empty())
        return g_ChannelFrames[idx];
    return g_ChannelFrames[static_cast<size_t>(ChannelCategory::Others)];
}

std::string PickChannelTopic(ChannelCategory cat)
{
    size_t idx = static_cast<size_t>(cat);
    if (idx >= 8 || g_ChannelTopics[idx].empty())
        return "";
    const std::vector<std::string>& pool = g_ChannelTopics[idx];
    return pool[pool.size() == 1 ? 0 : urand(0, pool.size() - 1)];
}

void AppendChannelMessage(const std::string& key, const std::string& sender, const std::string& text)
{
    if (key.empty() || text.empty())
        return;
    time_t now = time(nullptr);
    std::lock_guard<std::mutex> lock(g_ChannelThreadsMutex);
    ChannelThread& ct = g_ChannelThreads[key];
    ct.messages.push_back({ sender, text, now });
    ct.lastTouch = now;
    while (ct.messages.size() > g_ConversationThreadWindow)
        ct.messages.pop_front();
    // LRU backstop: cap the number of channel keys (never evict the key just touched).
    while (g_ChannelThreads.size() > g_ConversationThreadMaxChannels)
    {
        auto oldest = g_ChannelThreads.begin();
        for (auto it = g_ChannelThreads.begin(); it != g_ChannelThreads.end(); ++it)
            if (it->second.lastTouch < oldest->second.lastTouch)
                oldest = it;
        if (oldest->first == key)
            break;
        g_ChannelThreads.erase(oldest);
    }
}

std::string GetChannelThreadPrompt(const std::string& key)
{
    if (!g_EnableConversationThreading || key.empty())
        return "";
    time_t now = time(nullptr);
    time_t ttl = (time_t)g_ConversationThreadTTLMinutes * 60;
    std::lock_guard<std::mutex> lock(g_ChannelThreadsMutex);
    auto it = g_ChannelThreads.find(key);
    if (it == g_ChannelThreads.end())
        return "";
    std::string thread;
    for (const auto& m : it->second.messages)
    {
        if (now - m.ts > ttl)
            continue; // freshness filter: never mix in stale lines
        if (!thread.empty())
            thread += "\n";
        thread += m.sender + ": " + m.text;
    }
    if (thread.empty())
        return "";
    return SafeFormat(g_ConversationThreadTemplate, fmt::arg("thread", thread));
}

void CleanupChannelThreads()
{
    time_t now = time(nullptr);
    time_t ttl = (time_t)g_ConversationThreadTTLMinutes * 60;
    std::lock_guard<std::mutex> lock(g_ChannelThreadsMutex);
    for (auto it = g_ChannelThreads.begin(); it != g_ChannelThreads.end(); )
    {
        if (now - it->second.lastTouch > ttl)
            it = g_ChannelThreads.erase(it);
        else
            ++it;
    }
}

void SaveBotConversationHistoryToDB()
{
    std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);

    for (const auto& [botGuid, playerMap] : g_BotConversationHistory) {
        for (const auto& [playerGuid, history] : playerMap) {
            for (const auto& pair : history) {
                const std::string& playerMessage = pair.first;
                const std::string& botReply = pair.second;

                std::string escPlayerMsg = playerMessage;
                CharacterDatabase.EscapeString(escPlayerMsg);

                std::string escBotReply = botReply;
                CharacterDatabase.EscapeString(escBotReply);

                CharacterDatabase.Execute(SafeFormat(
                    "INSERT IGNORE INTO mod_ollama_chat_history (bot_guid, player_guid, timestamp, player_message, bot_reply) "
                    "VALUES ({}, {}, NOW(), '{}', '{}')",
                    botGuid, playerGuid, escPlayerMsg, escBotReply));
            }
        }
    }

    // Cleanup: keep only the N most recent entries per bot/player pair
    std::string cleanupQuery = R"SQL(
        WITH ranked_history AS (
            SELECT
                bot_guid,
                player_guid,
                timestamp,
                ROW_NUMBER() OVER (
                    PARTITION BY bot_guid, player_guid
                    ORDER BY timestamp DESC
                ) as rn
            FROM mod_ollama_chat_history
        )
        DELETE FROM mod_ollama_chat_history
        WHERE (bot_guid, player_guid, timestamp) IN (
            SELECT bot_guid, player_guid, timestamp
            FROM ranked_history
            WHERE rn > {}
        );
    )SQL";
    CharacterDatabase.Execute(SafeFormat(cleanupQuery, g_MaxConversationHistory));
}

// Called when a bot sends a message (random chatter or other bot-initiated messages)
// This triggers other bots to potentially reply
void ProcessBotChatMessage(Player* bot, const std::string& msg, ChatChannelSourceLocal sourceLocal, Channel* channel)
{
    if (!bot || msg.empty())
        return;
        
    // If channel is nullptr but this is a channel-type message, try to find the channel
    if (!channel && sourceLocal == SRC_GENERAL_LOCAL)
    {
        // Look up the General channel for this bot's faction
        std::string channelName = "General";
        ChannelMgr* cMgr = ChannelMgr::forTeam(bot->GetTeamId());
        if (cMgr)
        {
            channel = cMgr->GetChannel(channelName, bot);
            if (g_DebugEnabled)
            {
                if (channel)
                    LOG_INFO("server.loading", "[Ollama Chat] ProcessBotChatMessage: Found General channel for bot {}", bot->GetName());
                else
                    LOG_ERROR("server.loading", "[Ollama Chat] ProcessBotChatMessage: Could not find General channel for bot {}", bot->GetName());
            }
        }
    }
    
    // Validate that bot is actually in the relevant chat group before triggering replies
    bool canSendMessage = false;
    switch (sourceLocal)
    {
        case SRC_SAY_LOCAL:
        case SRC_YELL_LOCAL:
            // Distance checks will be applied during eligibility filtering
            canSendMessage = true;
            break;
            
        case SRC_GENERAL_LOCAL:
            // Must have a channel object
            canSendMessage = (channel != nullptr);
            if (!canSendMessage && g_DebugEnabled)
                LOG_ERROR("server.loading", "[Ollama Chat] Bot {} cannot send to General - no channel found", bot->GetName());
            break;
            
        case SRC_GUILD_LOCAL:
        case SRC_OFFICER_LOCAL:
            // Must be in a guild with at least one real player online
            if (bot->GetGuildId() != 0)
            {
                Guild* guild = sGuildMgr->GetGuildById(bot->GetGuildId());
                if (guild)
                {
                    // Check if any real (non-bot) players are online in this guild
                    bool hasRealPlayer = false;
                    for (auto const& pair : ObjectAccessor::GetPlayers())
                    {
                        Player* member = pair.second;
                        if (member && member->GetGuildId() == bot->GetGuildId())
                        {
                            if (!PlayerbotsMgr::instance().GetPlayerbotAI(member))
                            {
                                hasRealPlayer = true;
                                break;
                            }
                        }
                    }
                    canSendMessage = hasRealPlayer;
                    if (!canSendMessage && g_DebugEnabled)
                        LOG_INFO("server.loading", "[Ollama Chat] Bot {} cannot send to Guild - no real players online in guild", bot->GetName());
                }
                else
                {
                    canSendMessage = false;
                    if (g_DebugEnabled)
                        LOG_ERROR("server.loading", "[Ollama Chat] Bot {} cannot send to Guild - guild not found", bot->GetName());
                }
            }
            else
            {
                canSendMessage = false;
                if (g_DebugEnabled)
                    LOG_ERROR("server.loading", "[Ollama Chat] Bot {} cannot send to Guild - not in a guild", bot->GetName());
            }
            break;
            
        case SRC_PARTY_LOCAL:
        case SRC_RAID_LOCAL:
            // Must be in a group with at least one real player
            if (bot->GetGroup())
            {
                Group* group = bot->GetGroup();
                bool hasRealPlayer = false;
                for (GroupReference* ref = group->GetFirstMember(); ref; ref = ref->next())
                {
                    Player* member = ref->GetSource();
                    if (member && !PlayerbotsMgr::instance().GetPlayerbotAI(member))
                    {
                        hasRealPlayer = true;
                        break;
                    }
                }
                canSendMessage = hasRealPlayer;
                if (!canSendMessage && g_DebugEnabled)
                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} cannot send to Party - no real players in group", bot->GetName());
            }
            else
            {
                canSendMessage = false;
                if (g_DebugEnabled)
                    LOG_ERROR("server.loading", "[Ollama Chat] Bot {} cannot send to Party - not in a group", bot->GetName());
            }
            break;
            
        case SRC_WHISPER_LOCAL:
            // Whispers are handled separately
            canSendMessage = true;
            break;
            
        default:
            canSendMessage = true;
            break;
    }
    
    if (!canSendMessage)
    {
        if (g_DebugEnabled)
            LOG_ERROR("server.loading", "[Ollama Chat] Bot {} cannot send message to {} - validation failed", 
                    bot->GetName(), ChatChannelSourceLocalStr[sourceLocal]);
        return;
    }
        
    // Convert ChatChannelSourceLocal back to chat type for ProcessChat
    uint32_t type = 0;
    switch (sourceLocal)
    {
        case SRC_SAY_LOCAL: type = CHAT_MSG_SAY; break;
        case SRC_YELL_LOCAL: type = CHAT_MSG_YELL; break;
        case SRC_PARTY_LOCAL: type = CHAT_MSG_PARTY; break;
        case SRC_RAID_LOCAL: type = CHAT_MSG_RAID; break;
        case SRC_GUILD_LOCAL: type = CHAT_MSG_GUILD; break;
        case SRC_OFFICER_LOCAL: type = CHAT_MSG_OFFICER; break;
        case SRC_WHISPER_LOCAL: type = CHAT_MSG_WHISPER; break;
        case SRC_GENERAL_LOCAL: type = CHAT_MSG_CHANNEL; break;
        default: type = CHAT_MSG_SAY; break;
    }
    
    std::string mutableMsg = msg; // ProcessChat takes non-const reference
    uint32_t lang = bot->GetTeamId() == TEAM_ALLIANCE ? LANG_COMMON : LANG_ORCISH;
    
    // Call the main ProcessChat function with bot as sender
    PlayerBotChatHandler::ProcessChat(bot, type, lang, mutableMsg, sourceLocal, channel, nullptr);
}

std::string GetBotHistoryPrompt(uint64_t botGuid, uint64_t playerGuid, std::string playerMessage)
{
    if(!g_EnableChatHistory)
    {
        return "";
    }
    
    std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);

    std::string result;
    const auto botIt = g_BotConversationHistory.find(botGuid);
    if (botIt == g_BotConversationHistory.end())
        return result;
    const auto playerIt = botIt->second.find(playerGuid);
    if (playerIt == botIt->second.end())
        return result;

    Player* player = ObjectAccessor::FindPlayer(ObjectGuid(playerGuid));
    std::string playerName = player ? player->GetName() : "The player";

    result += SafeFormat(g_ChatHistoryHeaderTemplate, fmt::arg("player_name", playerName));

    for (const auto& entry : playerIt->second) {
        result += SafeFormat(g_ChatHistoryLineTemplate,
            fmt::arg("player_name", playerName),
            fmt::arg("player_message", entry.first),
            fmt::arg("bot_reply", entry.second)
        );
    }

    result += SafeFormat(g_ChatHistoryFooterTemplate,
        fmt::arg("player_name", playerName),
        fmt::arg("player_message", playerMessage)
    );

    return result;
}

// --- Helper: Spells ---
std::string ChatHandler_GetBotSpellInfo(Player* bot)
{
    if (!bot) return "";

    // Unique by spell name, keep the highest rank seen.
    std::map<std::string, uint32> bestRank;
    for (const auto& spellPair : bot->GetSpellMap())
    {
        uint32 spellId = spellPair.first;
        const SpellInfo* spellInfo = sSpellMgr->GetSpellInfo(spellId);
        if (!spellInfo || (spellInfo->Attributes & SPELL_ATTR0_PASSIVE)) continue;
        if (spellInfo->SpellFamilyName == SPELLFAMILY_GENERIC) continue;
        if (bot->HasSpellCooldown(spellId)) continue;
        const char* name = spellInfo->SpellName[0];
        if (!name || !*name) continue;

        uint32 rank = spellInfo->GetRank();
        auto it = bestRank.find(name);
        if (it == bestRank.end() || rank > it->second)
            bestRank[name] = rank;
    }
    if (bestRank.empty()) return "";

    // Take the N highest-rank spells as a "signature" proxy.
    std::vector<std::pair<uint32, std::string>> byRank;
    for (auto const& kv : bestRank) byRank.emplace_back(kv.second, kv.first);
    std::sort(byRank.begin(), byRank.end(),
        [](auto const& a, auto const& b) { return a.first > b.first; });

    std::string out = "You can call on ";
    uint32_t n = 0;
    for (auto const& s : byRank)
    {
        if (n >= g_SnapshotMaxSpells) break;
        if (n) out += ", ";
        out += s.second;
        ++n;
    }
    out += ".";
    return out;
}

// --- Helper: Group info ---
std::string ChatHandler_GetGroupStatus(Player* bot)
{
    if (!bot || !bot->GetGroup()) return "";
    Group* group = bot->GetGroup();

    std::vector<std::pair<float, std::string>> members;
    for (GroupReference* ref = group->GetFirstMember(); ref; ref = ref->next())
    {
        Player* member = ref->GetSource();
        if (!member || !member->GetMap() || member == bot) continue;

        std::string entry = member->GetName() + ", a " +
            FormatPlayerRace(member->getRace()) + " " + FormatPlayerClass(member->getClass());
        if (Unit* foe = member->GetVictim())
            entry += " battling a " + foe->GetName();

        members.emplace_back(bot->GetDistance(member), entry);
    }
    if (members.empty()) return "";

    std::sort(members.begin(), members.end(),
        [](auto const& a, auto const& b) { return a.first < b.first; });

    std::string out = "With you: ";
    uint32_t n = 0;
    for (auto const& m : members)
    {
        if (n >= g_SnapshotMaxGroup) break;
        if (n) out += "; ";
        out += m.second;
        ++n;
    }
    out += ".";
    return out;
}

// --- Helper: Visible players ---
std::string ChatHandler_GetVisiblePlayers(Player* bot, float radius = 40.0f)
{
    if (!bot || !bot->GetMap()) return "";

    std::vector<std::pair<float, std::string>> seen;
    for (auto const& pair : ObjectAccessor::GetPlayers())
    {
        Player* player = pair.second;
        if (!player || player == bot) continue;
        if (!player->IsInWorld() || player->IsGameMaster()) continue;
        if (player->GetMap() != bot->GetMap()) continue;
        if (!bot->IsWithinDistInMap(player, radius)) continue;
        if (!bot->IsWithinLOS(player->GetPositionX(), player->GetPositionY(), player->GetPositionZ())) continue;

        std::string faction = (player->GetTeamId() == TEAM_ALLIANCE ? "Alliance" : "Horde");
        seen.emplace_back(bot->GetDistance(player), player->GetName() + " the " + faction);
    }
    if (seen.empty()) return "";

    std::sort(seen.begin(), seen.end(),
        [](auto const& a, auto const& b) { return a.first < b.first; });

    std::string out = "Nearby: ";
    uint32_t n = 0;
    for (auto const& s : seen)
    {
        if (n >= g_SnapshotMaxPlayers) break;
        if (n) out += ", ";
        out += s.second;
        ++n;
    }
    out += ".";
    return out;
}

// --- Helper: Visible locations/objects (creatures only) ---
std::string ChatHandler_GetVisibleLocations(Player* bot, float radius = 40.0f)
{
    if (!bot || !bot->GetMap()) return "";
    Map* map = bot->GetMap();

    std::vector<std::pair<float, std::string>> seen;
    for (auto const& pair : map->GetCreatureBySpawnIdStore())
    {
        Creature* c = pair.second;
        if (!c || c->isDead()) continue;
        if (c->GetGUID() == bot->GetGUID()) continue;
        if (c->IsPet() || c->IsTotem()) continue;
        if (!bot->IsWithinDistInMap(c, radius)) continue;
        if (!bot->IsWithinLOS(c->GetPositionX(), c->GetPositionY(), c->GetPositionZ())) continue;
        if (c->GetName().empty()) continue;

        std::string entry = (c->IsHostileTo(bot) ? "a hostile " : "a ") + c->GetName();
        seen.emplace_back(bot->GetDistance(c), entry);
    }
    if (seen.empty()) return "";

    std::sort(seen.begin(), seen.end(),
        [](auto const& a, auto const& b) { return a.first < b.first; });

    std::string out = "You see ";
    uint32_t n = 0;
    for (auto const& s : seen)
    {
        if (n >= g_SnapshotMaxCreatures) break;
        if (n) out += ", ";
        out += s.second;
        ++n;
    }
    out += " nearby.";
    return out;
}

// --- Helper: Combat summary ---
std::string ChatHandler_GetCombatSummary(Player* bot)
{
    if (!bot) return "";

    float pct = bot->GetMaxHealth() > 0 ? bot->GetHealthPct() : 100.0f;
    std::string band;
    if (pct >= 85.0f)      band = "hale";
    else if (pct >= 40.0f) band = "wounded";
    else if (pct >= 15.0f) band = "badly wounded";
    else                   band = "near death";

    if (bot->IsInCombat())
    {
        if (Unit* victim = bot->GetVictim())
            return "You are " + band + ", fighting a " + victim->GetName() + ".";
        return "You are " + band + " and under attack.";
    }
    return "You are " + band + " and at rest.";
}


std::string GenerateBotGameStateSnapshot(Player* bot)
{
    if (!bot) return "";

    std::string combat  = ChatHandler_GetCombatSummary(bot);
    std::string group   = ChatHandler_GetGroupStatus(bot);
    std::string spells  = ChatHandler_GetBotSpellInfo(bot);
    std::string los     = ChatHandler_GetVisibleLocations(bot);
    std::string players = ChatHandler_GetVisiblePlayers(bot);

    // Active quests only (in progress / ready to turn in), titles kept, capped.
    std::string quests;
    uint32_t qn = 0;
    for (auto const& [questId, qsd] : bot->getQuestStatusMap())
    {
        if (qn >= g_SnapshotMaxQuests) break;

        std::string phrase;
        if (qsd.Status == QUEST_STATUS_INCOMPLETE)    phrase = "seeking to finish";
        else if (qsd.Status == QUEST_STATUS_COMPLETE) phrase = "ready to report";
        else continue;

        Quest const* quest = sObjectMgr->GetQuestTemplate(questId);
        if (!quest) continue;
        std::string title = quest->GetTitle();
        if (auto const* locale = sObjectMgr->GetQuestLocale(questId))
        {
            int locIdx = bot->GetSession()->GetSessionDbLocaleIndex();
            if (locIdx >= 0)
                ObjectMgr::GetLocaleString(locale->Title, locIdx, title);
        }
        if (title.empty()) continue;

        if (qn) quests += " ";
        quests += "You are " + phrase + " \"" + title + "\"";
        if (qsd.Status == QUEST_STATUS_COMPLETE) quests += " done";
        quests += ".";
        ++qn;
    }

    std::string snapshot = SafeFormat(
        g_ChatBotSnapshotTemplate,
        fmt::arg("combat", combat),
        fmt::arg("group", group),
        fmt::arg("spells", spells),
        fmt::arg("quests", quests),
        fmt::arg("los", los),
        fmt::arg("players", players)
    );

    if (g_DebugEnabled)
        LOG_INFO("server.loading", "[Ollama Chat] Snapshot for {}: {}", bot->GetName(), snapshot);

    return snapshot;
}


void PlayerBotChatHandler::ProcessChat(Player* player, uint32_t /*type*/, uint32_t lang, std::string& msg, ChatChannelSourceLocal sourceLocal, Channel* channel, Player* receiver)
{
    if (player == nullptr) {
        LOG_ERROR("server.loading", "[Ollama Chat] ProcessChat: player is null");
        return;
    }
    if (msg.empty()) {
        return;
    }
    if (lang == LANG_ADDON) return;

    if (g_EnableConversationThreading)
        AppendChannelMessage(ComputeChannelKey(sourceLocal, channel, player), player->GetName(), msg);
    std::string chanName = (channel != nullptr) ? channel->GetName() : "Unknown";
    uint32_t channelId = (channel != nullptr) ? channel->GetChannelId() : 0;
    std::string receiverName = (receiver != nullptr) ? receiver->GetName() : "None";
    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading",
                "[Ollama Chat] Player {} sent msg: '{}' | Source: {} | Channel Name: {} | Channel ID: {} | Receiver: {}",
                player->GetName(), msg, (int)sourceLocal, chanName, channelId, receiverName);
    }


    auto startsWithWord = [](const std::string& text, const std::string& word) {
        if (text.size() < word.size()) return false;
        if (text.compare(0, word.size(), word) != 0) return false;
        // If exact length match or next char is whitespace/punctuation, it's a word
        return text.size() == word.size() || !std::isalnum((unsigned char)text[word.size()]);
    };

    std::string trimmedMsg = rtrim(msg);
    for (const std::string& blacklist : g_BlacklistCommands)
    {
        if (startsWithWord(trimmedMsg, blacklist))
        {
            if (g_DebugEnabled)
                LOG_INFO("server.loading",
                         "[Ollama Chat] Message starts with '{}' (blacklisted). Skipping bot responses.",
                         blacklist);
            return;
        }
    }
    
    // Check if this channel type is disabled
    if (sourceLocal == SRC_GENERAL_LOCAL && g_DisableForCustomChannels)
    {
        if (g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Custom channels are disabled, skipping");
        }
        return;
    }
    
    if ((sourceLocal == SRC_SAY_LOCAL || sourceLocal == SRC_YELL_LOCAL) && g_DisableForSayYell)
    {
        if (g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Say/Yell channels are disabled, skipping");
        }
        return;
    }
    
    if ((sourceLocal == SRC_GUILD_LOCAL || sourceLocal == SRC_OFFICER_LOCAL) && g_DisableForGuild)
    {
        if (g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Guild channels are disabled, skipping");
        }
        return;
    }
    
    if ((sourceLocal == SRC_PARTY_LOCAL || sourceLocal == SRC_RAID_LOCAL) && g_DisableForParty)
    {
        if (g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Party/Raid channels are disabled, skipping");
        }
        return;
    }
             
    PlayerbotAI* senderAI = PlayerbotsMgr::instance().GetPlayerbotAI(player);
    bool senderIsBot = (senderAI && senderAI->IsBotAI());
    
    std::vector<Player*> eligibleBots;
    
    // Handle different chat sources differently
    if (sourceLocal == SRC_WHISPER_LOCAL && receiver != nullptr)
    {
        // Check if whisper replies are disabled
        if (!g_EnableWhisperReplies)
        {
            if(g_DebugEnabled)
            {
                LOG_INFO("server.loading", "[Ollama Chat] Whisper replies are disabled, skipping");
            }
            return;
        }
        
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Processing whisper from {} to {}", 
                    player->GetName(), receiver->GetName());
        }
        
        // Skip bot-to-bot whispers to prevent Ollama responses
        if (senderIsBot)
        {
            return;
        }
        
        // For whispers, only the receiver bot can respond (if it's a bot)
        PlayerbotAI* receiverAI = PlayerbotsMgr::instance().GetPlayerbotAI(receiver);
        if (receiverAI && receiverAI->IsBotAI())
        {
            eligibleBots.push_back(receiver);
            if(g_DebugEnabled)
            {
                LOG_INFO("server.loading", "[Ollama Chat] Found eligible bot {} for whisper", receiver->GetName());
            }
        }
        else if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Whisper target {} is not a bot or has no AI", receiver->GetName());
        }
    }
    else if (channel != nullptr)
    {
        // For channel chat, find all bots that are in the same channel instance
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Processing channel message in '{}' (ID: {})", 
                    channel->GetName(), channel->GetChannelId());
        }
        
        // Verify the original channel is valid before proceeding
        if (!channel)
        {
            if(g_DebugEnabled)
            {
                LOG_ERROR("server.loading", "[Ollama Chat] Channel is null, cannot process channel message");
            }
            return;
        }
        
        // For channel chat, simply find all bots in the same zone as the player
        auto const& allPlayers = ObjectAccessor::GetPlayers();
        for (auto const& itr : allPlayers)
        {
            Player* candidate = itr.second;
            if (!candidate || candidate == player)
                continue;
                
            // Skip non-bots early
            PlayerbotAI* candidateAI = PlayerbotsMgr::instance().GetPlayerbotAI(candidate);
            if (!candidateAI || !candidateAI->IsBotAI())
                continue;
            
            // Check if this is a local or global channel
            bool isLocalChannel = (channel->GetName().find("General -") != std::string::npos || 
                                  channel->GetName().find("Trade -") != std::string::npos ||
                                  channel->GetName().find("LocalDefense -") != std::string::npos);
            
            bool isGlobalChannel = (channel->GetName().find("World") != std::string::npos || channel->GetName().find("LookingForGroup") != std::string::npos);
        
            // For local channels, bot must be in same zone as player
            if (isLocalChannel)
            {
                // ZONE CHECK: Bot must be in exact same zone as player
                if (candidate->GetZoneId() != player->GetZoneId())
                {
                    if(g_DebugEnabled)
                    {
                        //LOG_ERROR("server.loading", "[Ollama Chat] Bot {} FAILED zone check - Bot zone: {}, Player zone: {}, Channel: '{}'", candidate->GetName(), candidate->GetZoneId(), player->GetZoneId(), channel->GetName());
                    }
                    continue; // SKIP this bot - wrong zone
                }
            }
            // For global channels like World, no zone restriction
            
            // CHANNEL MEMBERSHIP CHECK: Bot must actually be in the channel
            if (!candidate->IsInChannel(channel))
            {
                if(g_DebugEnabled)
                {
                    //LOG_INFO("server.loading", "[Ollama Chat] Bot {} not in channel '{}', skipping", candidate->GetName(), channel->GetName());
                }
                continue;
            }
            
            // FACTION CHECK: For non-global channels, ensure same faction
            if (candidate->GetTeamId() != player->GetTeamId())
            {
                if (!isGlobalChannel)
                {
                    if(g_DebugEnabled)
                    {
                        //LOG_ERROR("server.loading", "[Ollama Chat] Bot {} FAILED faction check - Bot: {}, Player: {}, Channel: '{}'", candidate->GetName(), (int)candidate->GetTeamId(), (int)player->GetTeamId(), channel->GetName());
                    }
                    continue; // SKIP this bot - wrong faction
                }
            }
            
            // CHANNEL MEMBERSHIP CHECK: Verify bot is actually in the channel
            if (!candidate->IsInChannel(channel))
            {
                if(g_DebugEnabled)
                {
                    //LOG_ERROR("server.loading", "[Ollama Chat] Bot {} FAILED channel membership check - Not in channel '{}'", candidate->GetName(), channel->GetName());
                }
                continue; // SKIP this bot - not in the channel
            }
            
            // REAL PLAYER CHECK: Channel must have at least one real player
            bool hasRealPlayerInChannel = false;
            for (auto const& playerItr : allPlayers)
            {
                Player* potentialRealPlayer = playerItr.second;
                if (potentialRealPlayer && potentialRealPlayer->IsInChannel(channel))
                {
                    PlayerbotAI* realPlayerAI = PlayerbotsMgr::instance().GetPlayerbotAI(potentialRealPlayer);
                    if (!realPlayerAI || !realPlayerAI->IsBotAI())
                    {
                        hasRealPlayerInChannel = true;
                        break;
                    }
                }
            }
            
            if (!hasRealPlayerInChannel)
            {
                if(g_DebugEnabled)
                {
                    //LOG_INFO("server.loading", "[Ollama Chat] Bot {} skipped - no real players in channel '{}'", candidate->GetName(), channel->GetName());
                }
                continue;
            }
            
            // ONLY add bots that passed ALL verifications
            eligibleBots.push_back(candidate);
            if(g_DebugEnabled)
            {
                // LOG_INFO("server.loading", "[Ollama Chat] VERIFIED eligible bot {} in channel '{}' - Distance: {:.2f}, Zone match: {}", candidate->GetName(), channel->GetName(), candidate->GetDistance(player), (candidate->GetZoneId() == player->GetZoneId()));
            }
        }
        
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Found {} bots in channel instance '{}'", 
                    eligibleBots.size(), channel->GetName());
        }
    }
    else
    {
        // For other chat types (say, yell, guild, party, etc.), use all players and filter by eligibility
        auto const& allPlayers = ObjectAccessor::GetPlayers();
        for (auto const& itr : allPlayers)
        {
            Player* candidate = itr.second;
            if (candidate->IsInWorld() && candidate != player)
            {
                PlayerbotAI* candidateAI = PlayerbotsMgr::instance().GetPlayerbotAI(candidate);
                if (candidateAI && candidateAI->IsBotAI())
                {
                    // For Guild/Party, verify there's a real player in that guild/party
                    if (sourceLocal == SRC_GUILD_LOCAL || sourceLocal == SRC_OFFICER_LOCAL)
                    {
                        if (candidate->GetGuildId() != 0)
                        {
                            // Check if any real player is online in this guild
                            bool hasRealPlayerInGuild = false;
                            for (auto const& guildPlayerItr : allPlayers)
                            {
                                Player* guildMember = guildPlayerItr.second;
                                if (guildMember && guildMember->GetGuildId() == candidate->GetGuildId())
                                {
                                    PlayerbotAI* memberAI = PlayerbotsMgr::instance().GetPlayerbotAI(guildMember);
                                    if (!memberAI || !memberAI->IsBotAI())
                                    {
                                        hasRealPlayerInGuild = true;
                                        break;
                                    }
                                }
                            }
                            if (!hasRealPlayerInGuild)
                                continue; // Skip bot - no real players in guild
                        }
                    }
                    else if (sourceLocal == SRC_PARTY_LOCAL || sourceLocal == SRC_RAID_LOCAL)
                    {
                        Group* group = candidate->GetGroup();
                        if (group)
                        {
                            // Check if any real player is in this group
                            bool hasRealPlayerInGroup = false;
                            for (GroupReference* ref = group->GetFirstMember(); ref; ref = ref->next())
                            {
                                Player* member = ref->GetSource();
                                if (member)
                                {
                                    PlayerbotAI* memberAI = PlayerbotsMgr::instance().GetPlayerbotAI(member);
                                    if (!memberAI || !memberAI->IsBotAI())
                                    {
                                        hasRealPlayerInGroup = true;
                                        break;
                                    }
                                }
                            }
                            if (!hasRealPlayerInGroup)
                                continue; // Skip bot - no real players in group
                        }
                    }
                    else if (sourceLocal == SRC_SAY_LOCAL || sourceLocal == SRC_YELL_LOCAL)
                    {
                        // For Say/Yell, require a real player within hearing distance
                        float threshold = (sourceLocal == SRC_SAY_LOCAL) ? g_SayDistance : g_YellDistance;
                        bool hasRealPlayerNearby = false;
                        
                        if (candidate->IsInWorld() && threshold > 0.0f)
                        {
                            for (auto const& nearbyPlayerItr : allPlayers)
                            {
                                Player* nearbyPlayer = nearbyPlayerItr.second;
                                if (nearbyPlayer && nearbyPlayer->IsInWorld())
                                {
                                    PlayerbotAI* nearbyAI = PlayerbotsMgr::instance().GetPlayerbotAI(nearbyPlayer);
                                    if (!nearbyAI || !nearbyAI->IsBotAI())
                                    {
                                        if (candidate->GetDistance(nearbyPlayer) <= threshold)
                                        {
                                            hasRealPlayerNearby = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        
                        if (!hasRealPlayerNearby)
                            continue; // Skip bot - no real player can hear Say/Yell
                    }
                    
                    eligibleBots.push_back(candidate);
                }
            }
        }
    }
    
    std::vector<Player*> candidateBots;
    int notEligibleCount = 0;
    for (Player* bot : eligibleBots)
    {
        if (!bot)
        {
            continue;
        }
        
        // For channel messages, bots in eligibleBots have already passed STRICT channel checks
        // Only run additional eligibility checks for non-channel sources
        // EXCEPTION: If channel is nullptr but sourceLocal is a channel type (like GENERAL), 
        // treat it as a channel message (happens with bot-initiated messages)
        bool isChannelSource = (sourceLocal == SRC_GENERAL_LOCAL);
        
        if (channel != nullptr || isChannelSource)
        {
            // Channel bots have already been verified to be in EXACT same channel instance
            // OR this is a channel-type source (General) even without channel object
            candidateBots.push_back(bot);
        }
        else
        {
            // For non-channel sources (Say/Yell/Guild/Party/Whisper), run the full eligibility check
            if (IsBotEligibleForChatChannelLocal(bot, player, sourceLocal, channel, receiver))
                candidateBots.push_back(bot);
            else
                notEligibleCount++;
        }
    }
    
    if (g_DebugEnabled && notEligibleCount > 0)
    {
        LOG_INFO("server.loading", "[Ollama Chat] {} bots not eligible for {} (distance/guild/party checks failed)", 
                notEligibleCount, ChatChannelSourceLocalStr[sourceLocal]);
    }
    
    // Determine reply chance based on channel type
    uint32_t chance;
    if (sourceLocal == SRC_SAY_LOCAL || sourceLocal == SRC_YELL_LOCAL)
    {
        // Say/Yell channel type
        chance = senderIsBot ? g_BotReplyChance_Say : g_PlayerReplyChance_Say;
    }
    else if (sourceLocal == SRC_PARTY_LOCAL || sourceLocal == SRC_RAID_LOCAL)
    {
        // Party/Raid channel type
        chance = senderIsBot ? g_BotReplyChance_Party : g_PlayerReplyChance_Party;
    }
    else if (sourceLocal == SRC_GUILD_LOCAL || sourceLocal == SRC_OFFICER_LOCAL)
    {
        // Guild/Officer channel type
        chance = senderIsBot ? g_BotReplyChance_Guild : g_PlayerReplyChance_Guild;
    }
    else if (sourceLocal == SRC_GENERAL_LOCAL)
    {
        // General/Trade/Custom channel type
        chance = senderIsBot ? g_BotReplyChance_Channel : g_PlayerReplyChance_Channel;
    }
    else
    {
        // Default fallback (whispers, etc.) - use Say chances
        chance = senderIsBot ? g_BotReplyChance_Say : g_PlayerReplyChance_Say;
    }
    
    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[Ollama Chat] Sender: {} ({}), Channel: {}, Reply Chance: {}%, Candidate Bots: {}",
                player->GetName(), senderIsBot ? "BOT" : "PLAYER", ChatChannelSourceLocalStr[sourceLocal], chance, candidateBots.size());
    }
    
    std::vector<Player*> finalCandidates;
    
    // For whispers, handle directly - there should only be one receiver bot
    if (sourceLocal == SRC_WHISPER_LOCAL)
    {
        if (!candidateBots.empty())
        {
            Player* whisperBot = candidateBots[0]; // Should only be one bot for whispers
            if (!(g_DisableRepliesInCombat && whisperBot->IsInCombat()))
            {
                finalCandidates.push_back(whisperBot);
                if(g_DebugEnabled)
                {
                    LOG_INFO("server.loading", "[Ollama Chat] Whisper: Bot {} selected to respond", whisperBot->GetName());
                }
            }
        }
    }
    else
    {
        // Handle non-whisper chats with normal multi-bot logic
        std::vector<std::pair<size_t, Player*>> mentionedBots;

        // Helper to convert string to lowercase safely
        auto toLowerStr = [](const std::string& str) -> std::string {
            std::string result = str;
            for (char& c : result)
            {
                c = std::tolower(static_cast<unsigned char>(c));
            }
            return result;
        };

        // Helper to check if a bot name is mentioned as a complete word
        auto isBotNameMentioned = [&trimmedMsg, &toLowerStr](const std::string& botName) -> size_t {
            std::string lowerMsg = toLowerStr(trimmedMsg);
            std::string lowerBotName = toLowerStr(botName);
            
            size_t pos = 0;
            while ((pos = lowerMsg.find(lowerBotName, pos)) != std::string::npos)
            {
                // Check if it's a word boundary before the name
                bool validStart = (pos == 0 || !std::isalnum(static_cast<unsigned char>(lowerMsg[pos - 1])));
                // Check if it's a word boundary after the name
                size_t endPos = pos + lowerBotName.length();
                bool validEnd = (endPos >= lowerMsg.length() || !std::isalnum(static_cast<unsigned char>(lowerMsg[endPos])));
                
                if (validStart && validEnd)
                {
                    return pos; // Found a valid word-boundary match
                }
                pos++; // Continue searching
            }
            return std::string::npos;
        };

        for (Player* bot : candidateBots)
        {
            if (!bot)
            {
                continue;
            }
            if (g_DisableRepliesInCombat && bot->IsInCombat())
            {
                continue;
            }
            
            size_t pos = isBotNameMentioned(bot->GetName());
            if (pos != std::string::npos)
            {
                mentionedBots.emplace_back(pos, bot);
                if(g_DebugEnabled)
                {
                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} mentioned at position {} in message", bot->GetName(), pos);
                }
            }
        }

        if (!mentionedBots.empty())
        {
            // Sort by position to get the first mentioned bot
            std::sort(mentionedBots.begin(), mentionedBots.end(),
                      [](const std::pair<size_t, Player*> &a, const std::pair<size_t, Player*> &b) { return a.first < b.first; });
            Player* chosen = mentionedBots.front().second;
            if (!(g_DisableRepliesInCombat && chosen->IsInCombat()))
            {
                finalCandidates.push_back(chosen);
                if(g_DebugEnabled)
                {
                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} selected (mentioned first at position {})", 
                            chosen->GetName(), mentionedBots.front().first);
                }
            }
        }
        else
        {
            for (Player* bot : candidateBots)
            {
                if (g_DisableRepliesInCombat && bot->IsInCombat())
                {
                    if(g_DebugEnabled)
                    {
                        LOG_INFO("server.loading", "[Ollama Chat] Bot {} skipped - in combat", bot->GetName());
                    }
                    continue;
                }
                uint32_t roll = urand(0, 99);
                if (roll < chance)
                {
                    finalCandidates.push_back(bot);
                    if(g_DebugEnabled)
                    {
                        LOG_INFO("server.loading", "[Ollama Chat] Bot {} PASSED chance roll ({} < {}%)", bot->GetName(), roll, chance);
                    }
                }
                else if(g_DebugEnabled)
                {
                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} FAILED chance roll ({} >= {}%)", bot->GetName(), roll, chance);
                }
            }
        }
    }

    
    if (finalCandidates.empty())
    {
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] *** NO BOTS RESPONDING *** to {} from {} in {} channel. "
                    "Eligible: {}, Candidates: {}, Final: 0, Chance: {}%",
                    senderIsBot ? "BOT" : "PLAYER", player->GetName(), ChatChannelSourceLocalStr[sourceLocal],
                    eligibleBots.size(), candidateBots.size(), chance);
            LOG_INFO("server.loading", "[Ollama Chat] No eligible bots found to respond to message '{}'. "
                    "Source: {}, Eligible bots: {}, Candidate bots: {}, Combat disabled: {}",
                    msg, ChatChannelSourceLocalStr[sourceLocal], eligibleBots.size(), 
                    candidateBots.size(), g_DisableRepliesInCombat);
        }
        return;
    }
    
    if (finalCandidates.size() > g_MaxBotsToPick)
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(finalCandidates.begin(), finalCandidates.end(), g);
        uint32_t countToPick = urand(1, g_MaxBotsToPick);
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Limiting {} bots to {} (MaxBotsToPick)", finalCandidates.size(), countToPick);
        }
        finalCandidates.resize(countToPick);
    }
    
    if(g_DebugEnabled && !finalCandidates.empty())
    {
        std::string botNames;
        for (Player* bot : finalCandidates)
        {
            if (!botNames.empty()) botNames += ", ";
            botNames += bot->GetName();
        }
        LOG_INFO("server.loading", "[Ollama Chat] *** {} BOTS RESPONDING *** to {} from {} in {}: [{}]",
                finalCandidates.size(), senderIsBot ? "BOT" : "PLAYER", player->GetName(),
                ChatChannelSourceLocalStr[sourceLocal], botNames);
    }
    
    uint64_t senderGuid = player->GetGUID().GetRawValue();
    
    for (Player* bot : finalCandidates)
    {
        float distance = player->GetDistance(bot);
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Bot {} (distance: {}) is set to respond.", bot->GetName(), distance);
        }
        if (bot == nullptr) {
            continue;
        }
        ChannelCategory channelCat = ClassifyChannel(sourceLocal, channel);
        std::string prompt = GenerateBotPrompt(bot, msg, player, channelCat);
        if (g_EnableConversationThreading)
            prompt += GetChannelThreadPrompt(ComputeChannelKey(sourceLocal, channel, player));
        uint64_t botGuid = bot->GetGUID().GetRawValue();
        
        std::thread([botGuid, senderGuid, prompt, sourceLocal, channelId = (channel ? channel->GetChannelId() : 0), channelName = (channel ? channel->GetName() : ""), msg]() {
            try {
                // Use the QueryManager to submit the query.
                auto responseFuture = SubmitQuery(prompt);
                if (!responseFuture.valid())
                {
                    return;
                }
                std::string response = responseFuture.get();

                // Reacquire pointers by GUID.
                Player* botPtr = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
                Player* senderPtr = ObjectAccessor::FindPlayer(ObjectGuid(senderGuid));
                if (!botPtr)
                {
                    if(g_DebugEnabled)
                    {
                        LOG_ERROR("server.loading", "[Ollama Chat] Failed to reacquire bot from GUID {}", botGuid);
                    }
                    return;
                }
                if (!senderPtr)
                {
                    if(g_DebugEnabled)
                    {
                        LOG_ERROR("server.loading", "[Ollama Chat] Failed to reacquire sender from GUID {}", senderGuid);
                    }
                    return;
                }
                if (response.empty())
                {
                    if(g_DebugEnabled)
                    {
                        LOG_INFO("server.loading", "[OllamaChat] Bot {} skipped reply due to API error", botPtr->GetName());
                    }
                    return;
                }
                PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(botPtr);
                if (!botAI)
                {
                    if(g_DebugEnabled)
                    {
                        LOG_ERROR("server.loading", "[Ollama Chat] No PlayerbotAI found for bot {}", botPtr->GetName());
                    }
                    return;
                }
                
                // Simulate typing delay if enabled
                if (g_EnableTypingSimulation)
                {
                    uint32_t delay = g_TypingSimulationBaseDelay + (response.length() * g_TypingSimulationDelayPerChar);
                    if (g_DebugEnabled)
                        LOG_INFO("server.loading", "[OllamaChat] Bot {} simulating typing delay: {}ms for {} characters", 
                                 botPtr->GetName(), delay, response.length());
                    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
                    
                    // Reacquire pointers after delay
                    botPtr = ObjectAccessor::FindPlayer(ObjectGuid(botGuid));
                    if (!botPtr) return;
                    botAI = PlayerbotsMgr::instance().GetPlayerbotAI(botPtr);
                    if (!botAI) return;
                    senderPtr = ObjectAccessor::FindPlayer(ObjectGuid(senderGuid));
                    if (!senderPtr) return;
                }

                ApplyChatEmote(botPtr, response);
                // Route the response.
                if (channelId != 0 && !channelName.empty())
                {
                    // For channels, get the channel instance for the bot's team
                    ChannelMgr* cMgr = ChannelMgr::forTeam(botPtr->GetTeamId());
                    if (cMgr)
                    {
                        Channel* targetChannel = cMgr->GetChannel(channelName, botPtr);
                        if (targetChannel)
                        {
                            if(g_DebugEnabled)
                            {
                                LOG_INFO("server.loading", "[Ollama Chat] Bot {} found channel '{}' (ID: {}), checking membership...", 
                                        botPtr->GetName(), channelName, targetChannel->GetChannelId());
                            }
                            
                            if (botPtr->IsInChannel(targetChannel))
                            {
                                if(g_DebugEnabled)
                                {
                                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} is confirmed in channel '{}', sending message...", 
                                            botPtr->GetName(), channelName);
                                }
                                targetChannel->Say(botPtr->GetGUID(), response, LANG_UNIVERSAL);
                                ProcessBotChatMessage(botPtr, response, SRC_GENERAL_LOCAL, targetChannel);
                                if(g_DebugEnabled)
                                {
                                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} responded in channel {}: {}", 
                                            botPtr->GetName(), channelName, response);
                                }
                            }
                            else
                            {
                                if(g_DebugEnabled)
                                {
                                    LOG_ERROR("server.loading", "[Ollama Chat] Bot {} NOT in channel '{}' according to IsInChannel check - skipping reply", 
                                                botPtr->GetName(), channelName);
                                }
                                // Don't fallback to Say - if bot isn't in the channel, don't reply at all
                            }
                        }
                        else
                        {
                            if(g_DebugEnabled)
                            {
                                LOG_ERROR("server.loading", "[Ollama Chat] Bot {} cannot find channel '{}' (ID: {}) for team {} - skipping reply", 
                                         botPtr->GetName(), channelName, channelId, (int)botPtr->GetTeamId());
                            }
                            // Don't fallback to Say - if channel doesn't exist, don't reply at all
                        }
                    }
                }
                else
                {
                    switch (sourceLocal)
                    {
                        case SRC_GUILD_LOCAL: 
                            botAI->SayToGuild(response);
                            ProcessBotChatMessage(botPtr, response, SRC_GUILD_LOCAL, nullptr);
                            break;
                        case SRC_OFFICER_LOCAL: 
                            botAI->SayToGuild(response);
                            ProcessBotChatMessage(botPtr, response, SRC_OFFICER_LOCAL, nullptr);
                            break;
                        case SRC_PARTY_LOCAL: 
                            botAI->SayToParty(response);
                            ProcessBotChatMessage(botPtr, response, SRC_PARTY_LOCAL, nullptr);
                            break;
                        case SRC_RAID_LOCAL:  
                            botAI->SayToRaid(response);
                            ProcessBotChatMessage(botPtr, response, SRC_RAID_LOCAL, nullptr);
                            break;
                        case SRC_SAY_LOCAL:
                            // Only send Say if someone (real player or bot) is within say distance
                            {
                                bool someoneCanHear = false;
                                if (botPtr->IsInWorld())
                                {
                                    for (auto const& pair : ObjectAccessor::GetPlayers())
                                    {
                                        Player* nearbyPlayer = pair.second;
                                        if (nearbyPlayer && nearbyPlayer != botPtr && nearbyPlayer->IsInWorld())
                                        {
                                            if (botPtr->GetDistance(nearbyPlayer) <= g_SayDistance)
                                            {
                                                someoneCanHear = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                                
                                if (someoneCanHear)
                                {
                                    botAI->Say(response);
                                    ProcessBotChatMessage(botPtr, response, SRC_SAY_LOCAL, nullptr);
                                }
                                else if (g_DebugEnabled)
                                {
                                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} skipping Say reply - no one within {} yards to hear it", 
                                            botPtr->GetName(), g_SayDistance);
                                }
                            }
                            break;
                        case SRC_YELL_LOCAL:
                            // Only send Yell if someone is within yell distance
                            {
                                bool someoneCanHear = false;
                                if (botPtr->IsInWorld())
                                {
                                    for (auto const& pair : ObjectAccessor::GetPlayers())
                                    {
                                        Player* nearbyPlayer = pair.second;
                                        if (nearbyPlayer && nearbyPlayer != botPtr && nearbyPlayer->IsInWorld())
                                        {
                                            if (botPtr->GetDistance(nearbyPlayer) <= g_YellDistance)
                                            {
                                                someoneCanHear = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                                
                                if (someoneCanHear)
                                {
                                    botAI->Yell(response);
                                    ProcessBotChatMessage(botPtr, response, SRC_YELL_LOCAL, nullptr);
                                }
                                else if (g_DebugEnabled)
                                {
                                    LOG_INFO("server.loading", "[Ollama Chat] Bot {} skipping Yell reply - no one within {} yards to hear it", 
                                            botPtr->GetName(), g_YellDistance);
                                }
                            }
                            break;
                        case SRC_WHISPER_LOCAL:
                            // For whispers, find the original sender and whisper back
                            {
                                Player* originalSender = ObjectAccessor::FindPlayer(ObjectGuid(senderGuid));
                                if (originalSender)
                                {
                                    if(g_DebugEnabled)
                                    {
                                        LOG_INFO("server.loading", "[Ollama Chat] Bot {} whispering response '{}' to {}", 
                                                botPtr->GetName(), response, originalSender->GetName());
                                    }
                                    botAI->Whisper(response, originalSender->GetName());
                                    // Don't trigger ProcessBotChatMessage for whispers - they're private
                                }
                                else if(g_DebugEnabled)
                                {
                                    LOG_ERROR("server.loading", "[Ollama Chat] Cannot whisper response - original sender not found for GUID {}", senderGuid);
                                }
                            }
                            break;
                        default:              
                            botAI->Say(response);
                            ProcessBotChatMessage(botPtr, response, SRC_SAY_LOCAL, nullptr);
                            break;
                    }
                }
                
                // Update sentiment based on the player's message
                UpdateBotPlayerSentiment(botPtr, senderPtr, msg);
                
                AppendBotConversation(botGuid, senderGuid, msg, response);
                AppendBotRecentReply(botGuid, response);
                LogCrossBotRepetition(botPtr, response);
                if (botPtr->IsInWorld() && senderPtr->IsInWorld())
                {
                    float respDistance = senderPtr->GetDistance(botPtr);
                    if(g_DebugEnabled)
                    {
                        LOG_INFO("server.loading", "[Ollama Chat] Bot {} (distance: {}) responded: {}", botPtr->GetName(), respDistance, response);
                    }
                }
                else
                {
                    if(g_DebugEnabled)
                    {
                        LOG_INFO("server.loading", "[Ollama Chat] Bot {} responded: {} (distance not calculated - players not in world)", botPtr->GetName(), response);
                    }
                }
            }
            catch (const std::exception& ex)
            {
                if(g_DebugEnabled)
                {
                    LOG_ERROR("server.loading", "[Ollama Chat] Exception in bot response thread: {}", ex.what());
                }
            }
        }).detach();

    }
}

static bool IsBotEligibleForChatChannelLocal(Player* bot, Player* player, ChatChannelSourceLocal source, Channel* channel, Player* receiver)
{
    if (!bot || !player || bot == player)
    {
        if (g_DebugEnabled)
            LOG_INFO("server.loading", "[Ollama Chat] IsBotEligible: FAILED basic check - bot={}, player={}, same={}", 
                    (void*)bot, (void*)player, (bot == player));
        return false;
    }
    if (!PlayerbotsMgr::instance().GetPlayerbotAI(bot))
    {
        if (g_DebugEnabled)
            LOG_INFO("server.loading", "[Ollama Chat] IsBotEligible: Bot {} FAILED - no PlayerbotAI", bot->GetName());
        return false;
    }
        
    // For whispers, only the specific receiver should respond
    if (source == SRC_WHISPER_LOCAL)
    {
        // Don't allow bot-to-bot whisper responses
        PlayerbotAI* senderAI = PlayerbotsMgr::instance().GetPlayerbotAI(player);
        if (senderAI && senderAI->IsBotAI())
        {
            return false;
        }
        
        return (receiver && bot == receiver);
    }
    
    // Check team compatibility for non-proximity chats (except channels which can be cross-faction)
    // Say and Yell are proximity-based and don't require same faction
    bool isProximityChatSource = (source == SRC_SAY_LOCAL || source == SRC_YELL_LOCAL);
    if (!channel && !isProximityChatSource && bot->GetTeamId() != player->GetTeamId())
        return false;
    
    // For channels, check if bot is in the specific channel instance
    if (channel)
    {
        // Verify the channel is valid before proceeding
        if (!channel)
        {
            if(g_DebugEnabled)
            {
                LOG_ERROR("server.loading", "[Ollama Chat] IsBotEligibleForChatChannelLocal: Channel is null");
            }
            return false;
        }
            
        // ONLY use exact channel instance check - NO Player::IsInChannel() anymore
        ChannelMgr* candidateCMgr = ChannelMgr::forTeam(bot->GetTeamId());
        if (!candidateCMgr)
            return false;
            
        Channel* candidateChannel = candidateCMgr->GetChannel(channel->GetName(), bot);
        // Verify both channels are valid and are the exact same instance
        if (!candidateChannel || candidateChannel != channel)
        {
            if(g_DebugEnabled)
            {
                LOG_INFO("server.loading", "[Ollama Chat] IsBotEligibleForChatChannelLocal: Bot {} not in same channel instance '{}' - Bot team: {}, Channel ptr: {} vs {}", 
                        bot->GetName(), channel->GetName(), (int)bot->GetTeamId(),
                        (void*)candidateChannel, (void*)channel);
            }
            return false;
        }
        
        // Additional team check for cross-faction channels - only allow same faction unless it's a global channel
        if (bot->GetTeamId() != player->GetTeamId())
        {
            // Allow cross-faction only for specific global channels
            bool isGlobalChannel = (channel->GetName().find("World") != std::string::npos || 
                                   channel->GetName().find("LookingForGroup") != std::string::npos);
            if (!isGlobalChannel)
            {
                if(g_DebugEnabled)
                {
                    LOG_INFO("server.loading", "[Ollama Chat] IsBotEligibleForChatChannelLocal: Bot {} different faction from player - Bot: {}, Player: {}, Channel: '{}'", bot->GetName(), (int)bot->GetTeamId(), (int)player->GetTeamId(), channel->GetName());
                }
                return false;
            }
        }
    }
    
    bool isInParty = (player->GetGroup() && bot->GetGroup() && (player->GetGroup() == bot->GetGroup()));
    float threshold = 0.0f;
    
    switch (source)
    {
        case SRC_SAY_LOCAL:    
            threshold = g_SayDistance;
            if (threshold > 0.0f)
            {
                if (!bot->IsInWorld() || !player->IsInWorld())
                    return false;
                    
                float distance = bot->GetDistance(player);
                return distance <= threshold;
            }
            return false;
            
        case SRC_YELL_LOCAL:   
            threshold = g_YellDistance;
            return (threshold > 0.0f && player->GetDistance(bot) <= threshold);
            
        case SRC_GUILD_LOCAL:
        case SRC_OFFICER_LOCAL:
            return (player->GetGuild() && bot->GetGuildId() == player->GetGuildId());
            
        case SRC_PARTY_LOCAL:
        case SRC_RAID_LOCAL:
            return isInParty;
            
        case SRC_WHISPER_LOCAL:
            // For whispers, the bot should only respond if it's the specific receiver
            return (receiver && bot == receiver);
            
        case SRC_GENERAL_LOCAL:
            // For channels like General, Trade, etc., no distance check - only channel membership matters
            // Channel membership was already checked above
            return true;
            
        default:
            return false;
    }
}

std::string GenerateBotPrompt(Player* bot, std::string playerMessage, Player* player, ChannelCategory channelCat)
{
    if (!bot || !player) {
        return "";
    }
    PlayerbotAI* botAI = PlayerbotsMgr::instance().GetPlayerbotAI(bot);
    if (botAI == nullptr) {
        return "";
    }
    ChatHelper* helper = botAI->GetChatHelper();
    if (helper == nullptr) {
        return "";
    }
    if (g_ChatPromptTemplate.empty()) {
        LOG_ERROR("server.loading", "[Ollama Chat] GenerateBotPrompt: template is empty");
        return "";
    }

    AreaTableEntry const* botCurrentArea = botAI->GetCurrentArea();
    AreaTableEntry const* botCurrentZone = botAI->GetCurrentZone();

    uint64_t botGuid                = bot->GetGUID().GetRawValue();
    uint64_t playerGuid             = player->GetGUID().GetRawValue();

    std::string personality         = GetBotPersonality(bot);
    std::string personalityPrompt   = GetPersonalityPromptAddition(personality);
    std::string botName             = bot->GetName();
    uint32_t botLevel               = bot->GetLevel();
    uint8_t botGenderByte           = bot->getGender();
    std::string botAreaName         = botCurrentArea ? botAI->GetLocalizedAreaName(botCurrentArea): "UnknownArea";
    std::string botZoneName         = botCurrentZone ? botAI->GetLocalizedAreaName(botCurrentZone): "UnknownZone";
    std::string botMapName          = bot->GetMap() ? bot->GetMap()->GetMapName() : "UnknownMap";
    std::string botClass            = botAI->GetChatHelper()->FormatClass(bot->getClass());
    std::string botRace             = botAI->GetChatHelper()->FormatRace(bot->getRace());
    std::string botRole             = ChatHelper::FormatClass(bot, AiFactory::GetPlayerSpecTab(bot));
    std::string botGender           = (botGenderByte == 0 ? "Male" : "Female");
    std::string botFaction          = (bot->GetTeamId() == TEAM_ALLIANCE ? "Alliance" : "Horde");
    std::string botGuild            = (bot->GetGuild() ? bot->GetGuild()->GetName() : "No Guild");
    std::string botGroupStatus      = (bot->GetGroup() ? "In a group" : "Solo");
    uint32_t botGold                = bot->GetMoney() / 10000;

    std::string playerName          = player->GetName();
    uint32_t playerLevel            = player->GetLevel();
    std::string playerClass         = botAI->GetChatHelper()->FormatClass(player->getClass());
    std::string playerRace          = botAI->GetChatHelper()->FormatRace(player->getRace());
    std::string playerRole          = ChatHelper::FormatClass(player, AiFactory::GetPlayerSpecTab(player));
    uint8_t playerGenderByte        = player->getGender();
    std::string playerGender        = (playerGenderByte == 0 ? "Male" : "Female");
    std::string playerFaction       = (player->GetTeamId() == TEAM_ALLIANCE ? "Alliance" : "Horde");
    std::string playerGuild         = (player->GetGuild() ? player->GetGuild()->GetName() : "No Guild");
    std::string playerGroupStatus   = (player->GetGroup() ? "In a group" : "Solo");
    uint32_t playerGold             = player->GetMoney() / 10000;
    float playerDistance            = player->IsInWorld() && bot->IsInWorld() ? player->GetDistance(bot) : -1.0f;

    std::string chatHistory         = GetBotHistoryPrompt(botGuid, playerGuid, playerMessage);
    std::string sentimentInfo       = GetSentimentPromptAddition(bot, player);

    // Retrieve RAG information if enabled
    std::string ragInfo;
    if (g_EnableRAG && g_RAGSystem) {
        auto ragResults = g_RAGSystem->RetrieveRelevantInfo(playerMessage, g_RAGMaxRetrievedItems, g_RAGSimilarityThreshold);
        std::string ragContent = g_RAGSystem->GetFormattedRAGInfo(ragResults);
        if (!ragContent.empty()) {
            ragInfo = SafeFormat(g_RAGPromptTemplate, fmt::arg("rag_info", ragContent));
        }
        if (g_DebugEnabled) {
            LOG_INFO("server.loading", "[Ollama Chat] RAG Debug - Enabled: {}, System: {}, Message: '{}', Results: {}, Content length: {}",
                g_EnableRAG, (void*)g_RAGSystem, playerMessage, ragResults.size(), ragContent.length());
        }
    } else if (g_DebugEnabled) {
        LOG_INFO("server.loading", "[Ollama Chat] RAG Debug - Not enabled or no system - Enabled: {}, System: {}",
            g_EnableRAG, (void*)g_RAGSystem);
    }

    std::string extraInfo = SafeFormat(
        g_ChatExtraInfoTemplate,
        fmt::arg("bot_race", botRace),
        fmt::arg("bot_gender", botGender),
        fmt::arg("bot_role", botRole),
        fmt::arg("bot_faction", botFaction),
        fmt::arg("bot_guild", botGuild),
        fmt::arg("bot_group_status", botGroupStatus),
        fmt::arg("bot_gold", botGold),
        fmt::arg("player_race", playerRace),
        fmt::arg("player_gender", playerGender),
        fmt::arg("player_role", playerRole),
        fmt::arg("player_faction", playerFaction),
        fmt::arg("player_guild", playerGuild),
        fmt::arg("player_group_status", playerGroupStatus),
        fmt::arg("player_gold", playerGold),
        fmt::arg("player_distance", playerDistance),
        fmt::arg("bot_area", botAreaName),
        fmt::arg("bot_zone", botZoneName),
        fmt::arg("bot_map", botMapName)
    );
    
    std::string prompt = SafeFormat(
        g_ChatPromptTemplate,
        fmt::arg("bot_name", botName),
        fmt::arg("bot_level", botLevel),
        fmt::arg("bot_class", botClass),
        fmt::arg("bot_personality", personalityPrompt),
        fmt::arg("bot_personality_name", personality),
        fmt::arg("player_level", playerLevel),
        fmt::arg("player_class", playerClass),
        fmt::arg("player_name", playerName),
        fmt::arg("player_message", playerMessage),
        fmt::arg("extra_info", extraInfo),
        fmt::arg("chat_history", chatHistory),
        fmt::arg("sentiment_info", sentimentInfo)
    );

    // Add RAG information to the prompt if available
    if (!ragInfo.empty()) {
        prompt += ragInfo + "\n";
    }

    if(g_EnableChatBotSnapshotTemplate)
    {
        prompt += GenerateBotGameStateSnapshot(bot);
        if (g_EnableAntiRepetition)
            prompt += GetAntiRepetitionPrompt(bot->GetGUID().GetRawValue());
    }

    if (g_EnableCrossBotAntiRepetition)
        prompt += GetNearbyBotsRecentReplies(bot);

    if (g_EnableChannelFrames)
        prompt += " [" + GetChannelFrame(channelCat) + "]";

    prompt += BuildEmoteChatInstruction();

    // Debug logging for full prompt including RAG information
    if (g_DebugEnabled && g_DebugShowFullPrompt) {
        LOG_INFO("server.loading", "[Ollama Chat] Full prompt sent to bot {} for player {}: {}", botName, playerName, prompt);
    }

    return prompt;
}
