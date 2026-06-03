#ifndef MOD_OLLAMA_CHAT_HANDLER_H
#define MOD_OLLAMA_CHAT_HANDLER_H

#include "ScriptMgr.h"
#include <string>

enum ChatChannelSourceLocal
{
    SRC_UNDEFINED_LOCAL  = 0,
    SRC_SAY_LOCAL        = 1,
    SRC_PARTY_LOCAL      = 2,
    SRC_RAID_LOCAL       = 3,
    SRC_GUILD_LOCAL      = 4,
    SRC_OFFICER_LOCAL    = 5,
    SRC_YELL_LOCAL       = 6,
    SRC_WHISPER_LOCAL    = 7,
    SRC_GENERAL_LOCAL    = 17
};

// --------------------------------------------
// Per-channel conversation frames
// Ordinals are an ABI with the config arrays (g_ChannelFrames/Topics/Weights):
//   Guild=0 Party=1 Raid=2 Say=3 Yell=4 General=5 Trade=6 Others=7
// --------------------------------------------
enum class ChannelCategory : uint8_t
{
    Guild = 0, Party = 1, Raid = 2, Say = 3,
    Yell = 4, General = 5, Trade = 6, Others = 7
};

ChannelCategory ClassifyChannel(ChatChannelSourceLocal sourceLocal, Channel* channel);
std::string     GetChannelFrame(ChannelCategory cat);
std::string     PickChannelTopic(ChannelCategory cat);

extern const char* ChatChannelSourceLocalStr[];

std::string rtrim(const std::string& s);
ChatChannelSourceLocal GetChannelSourceLocal(uint32_t type);
void ProcessBotChatMessage(Player* bot, const std::string& msg, ChatChannelSourceLocal sourceLocal, Channel* channel);

void SaveBotConversationHistoryToDB();
void CleanupChannelThreads();

class PlayerBotChatHandler : public PlayerScript
{
public:
    PlayerBotChatHandler() : PlayerScript("PlayerBotChatHandler", {
        PLAYERHOOK_CAN_PLAYER_USE_CHAT,
        PLAYERHOOK_CAN_PLAYER_USE_PRIVATE_CHAT,
        PLAYERHOOK_CAN_PLAYER_USE_GROUP_CHAT,
        PLAYERHOOK_CAN_PLAYER_USE_GUILD_CHAT,
        PLAYERHOOK_CAN_PLAYER_USE_CHANNEL_CHAT,
    }) {}
    bool OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Player* receiver);
    bool OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg);
    bool OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Group* group);
    bool OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Guild* guild);
    bool OnPlayerCanUseChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, Channel* channel);

    static void ProcessChat(Player* player, uint32_t type, uint32_t lang, std::string& msg, ChatChannelSourceLocal sourceLocal, Channel* channel = nullptr, Player* receiver = nullptr);
};

std::string GenerateBotGameStateSnapshot(Player* bot);
void AppendBotRecentReply(uint64_t botGuid, const std::string& reply);
std::string GetAntiRepetitionPrompt(uint64_t botGuid);
std::string GetNearbyBotsRecentReplies(Player* speaker);
void LogCrossBotRepetition(Player* speaker, const std::string& emittedLine);

// Build a RAG query string from a bot's context: "<zone> <area> <faction> <class> <race>".
// Returns empty if the bot has no PlayerbotAI. Unknown zone/area contribute nothing
// (no "UnknownZone" junk tokens).
std::string BuildBotContextQuery(Player* bot);

#endif // MOD_OLLAMA_CHAT_HANDLER_H
