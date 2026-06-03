#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_sentiment.h"
#include "mod-ollama-chat_rag.h"
#include "Config.h"
#include "Log.h"
#include "Player.h"
#include "SharedDefines.h"
#include "mod-ollama-chat_api.h"
#include <fmt/core.h>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <set>
#include <unordered_map>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <functional>


// --------------------------------------------
// Distance/Range Configuration
// --------------------------------------------
float      g_SayDistance       = 30.0f;
float      g_YellDistance      = 100.0f;
float      g_RandomChatterRealPlayerDistance = 40.0f;
float      g_EventChatterRealPlayerDistance = 40.0f;

// --------------------------------------------
// Bot/Player Chatter Probability & Limits
// --------------------------------------------
// Per-channel-type reply chances
uint32_t   g_PlayerReplyChance_Say     = 90;
uint32_t   g_BotReplyChance_Say        = 10;
uint32_t   g_PlayerReplyChance_Channel = 50;
uint32_t   g_BotReplyChance_Channel    = 5;
uint32_t   g_PlayerReplyChance_Party   = 90;
uint32_t   g_BotReplyChance_Party      = 10;
uint32_t   g_PlayerReplyChance_Guild   = 70;
uint32_t   g_BotReplyChance_Guild      = 5;

uint32_t   g_MaxBotsToPick     = 2;
uint32_t   g_RandomChatterBotCommentChance   = 5;
uint32_t   g_RandomChatterMaxBotsPerPlayer   = 2;
uint32_t   g_EventChatterBotCommentChance    = 15;
uint32_t   g_EventChatterBotSelfCommentChance = 5;
uint32_t   g_EventChatterMaxBotsPerPlayer    = 2;

// --------------------------------------------
// Ollama LLM API Configuration
// --------------------------------------------
std::string g_OllamaUrl        = "http://localhost:11434/api/generate";
std::string g_OllamaModel      = "llama3.2:1b";
uint32_t    g_OllamaNumPredict = 40;
bool        g_SoftStopEnable = true;
float       g_OllamaTemperature = 0.8f;
float       g_OllamaTopP = 0.95f;
float       g_OllamaRepeatPenalty = 1.1f;
uint32_t    g_OllamaTopK = 0;
float       g_OllamaMinP = 0.0f;
float       g_OllamaPresencePenalty = 0.0f;
float       g_OllamaFrequencyPenalty = 0.0f;
uint32_t    g_OllamaNumCtx = 0;
uint32_t    g_OllamaNumThreads = 0;
std::string g_OllamaStop = "";
std::string g_OllamaSystemPrompt = "";
std::string g_OllamaSeed = "";

// --------------------------------------------
// Concurrency/Queueing
// --------------------------------------------
uint32_t    g_MaxConcurrentQueries = 0;

// --------------------------------------------
// Feature Toggles & Core Settings
// --------------------------------------------
bool        g_Enable                          = true;
bool        g_DisableRepliesInCombat          = true;
bool        g_EnableRandomChatter             = true;
bool        g_EnableEventChatter              = true;
bool        g_EnableRPPersonalities           = false;
bool        g_EnableWhisperReplies            = false;
bool        g_DebugEnabled                    = false;
bool        g_DebugShowFullPrompt             = false;

// --------------------------------------------
// Think Mode Support
// --------------------------------------------
bool g_ThinkModeEnableForModule = false;

// --------------------------------------------
// Random Chatter Timing
// --------------------------------------------
uint32_t    g_MinRandomInterval               = 45;
uint32_t    g_MaxRandomInterval               = 180;

// --------------------------------------------
// Conversation History Settings
// --------------------------------------------
uint32_t    g_MaxConversationHistory          = 5;
uint32_t    g_ConversationHistorySaveInterval = 10;

// --------------------------------------------
// Prompt Templates
// --------------------------------------------
std::string g_RandomChatterPromptTemplate;
std::vector<std::string> g_RandomChatterPromptVariations;
std::vector<std::string> g_RandomChatterQuestionVariations;
std::string g_EventChatterPromptTemplate;
std::string g_ChatPromptTemplate;
std::string g_ChatExtraInfoTemplate;

// --------------------------------------------
// Personality and Prompt Data
// --------------------------------------------
std::unordered_map<uint64_t, std::string> g_BotPersonalityList;
std::unordered_map<std::string, std::string> g_PersonalityPrompts;
std::vector<std::string> g_PersonalityKeys;
std::vector<std::string> g_PersonalityKeysRandomOnly;
std::string g_DefaultPersonalityPrompt;

// --------------------------------------------
// Chat History Templates and Toggles
// --------------------------------------------
bool        g_EnableChatHistory = true;
std::string g_ChatHistoryHeaderTemplate;
std::string g_ChatHistoryLineTemplate;
std::string g_ChatHistoryFooterTemplate;

// --------------------------------------------
// Chatbot Snapshot Template
// --------------------------------------------
bool        g_EnableChatBotSnapshotTemplate  = false;
std::string g_ChatBotSnapshotTemplate;
uint32_t    g_SnapshotMaxQuests    = 3;
uint32_t    g_SnapshotMaxGroup     = 4;
uint32_t    g_SnapshotMaxCreatures = 2;
uint32_t    g_SnapshotMaxPlayers   = 2;
uint32_t    g_SnapshotMaxSpells    = 2;

// --------------------------------------------
// Conversation History Store and Mutex
// --------------------------------------------
std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::deque<std::pair<std::string, std::string>>>> g_BotConversationHistory;
std::mutex g_ConversationHistoryMutex;

bool        g_EnableAntiRepetition = true;
uint32_t    g_AntiRepetitionWindow = 6;
std::string g_AntiRepetitionTemplate;
bool        g_EnableCrossBotAntiRepetition   = false;  // opt-in
uint32_t    g_CrossBotAntiRepetitionWindow   = 2;
uint32_t    g_CrossBotAntiRepetitionMaxLines = 6;
std::string g_CrossBotAntiRepetitionTemplate;
std::unordered_map<uint64_t, std::deque<std::string>> g_BotRecentReplies;
std::mutex  g_RecentRepliesMutex;

bool        g_EnableConversationThreading = true;
uint32_t    g_ConversationThreadWindow = 8;
uint32_t    g_ConversationThreadTTLMinutes = 30;
uint32_t    g_ConversationThreadMaxChannels = 64;
std::string g_ConversationThreadTemplate;
std::unordered_map<std::string, ChannelThread> g_ChannelThreads;
std::mutex  g_ChannelThreadsMutex;

bool                     g_EnableChannelFrames = true;
bool                     g_EnableChannelTopics = true;
std::string              g_ChannelFrames[8];
std::vector<std::string> g_ChannelTopics[8];
uint32_t                 g_ChannelWeights[8] = { 25, 20, 15, 30, 5, 15, 0, 0 };

bool                     g_ActivityChatterEnable = true;
uint32_t                 g_ActivityChatterChance = 40;
std::vector<std::string> g_ActivityTopicsSocial;
std::vector<std::string> g_ActivityTopicsFishing;
std::vector<std::string> g_ActivityTopicsGather;
std::vector<std::string> g_ActivityTopicsCraft;
std::vector<std::string> g_ActivityTopicsDuel;

time_t g_LastHistorySaveTime = 0;

// --------------------------------------------
// Bot-Player Sentiment Tracking System
// --------------------------------------------
bool        g_EnableSentimentTracking = true;
float       g_SentimentDefaultValue = 0.5f;              // Default sentiment value (0.5 = neutral)
float       g_SentimentAdjustmentStrength = 0.1f;        // How much to adjust sentiment per message
uint32_t    g_SentimentSaveInterval = 10;                // How often to save sentiment to DB (minutes)
std::string g_SentimentAnalysisPrompt = "Analyze the sentiment of this message: \"{message}\". Respond only with: POSITIVE, NEGATIVE, or NEUTRAL.";
std::string g_SentimentPromptTemplate = "Your relationship sentiment with {player_name} is {sentiment_value} (0.0=hostile, 0.5=neutral, 1.0=friendly). Use this to guide your tone and response.";

// In-memory sentiment storage and mutex
std::unordered_map<uint64_t, std::unordered_map<uint64_t, float>> g_BotPlayerSentiments;
std::mutex g_SentimentMutex;
time_t g_LastSentimentSaveTime = 0;

// --------------------------------------------
// RAG (Retrieval-Augmented Generation) System
// --------------------------------------------
bool        g_EnableRAG = false;
std::string g_RAGDataPath = "rag/";
uint32_t    g_RAGMaxRetrievedItems = 3;
float       g_RAGSimilarityThreshold = 0.3f;
std::string g_RAGPromptTemplate;
bool        g_RAGExpandReferences = true;
uint32_t    g_RAGMaxReferences = 5;
bool        g_RAGRandomizeSelection = true;
uint32_t    g_RAGSelectionPoolSize = 8;
bool        g_RAGImprovedScoring = true;
bool        g_EnableRAGInitiated    = true;
uint32_t    g_RAGInitiatedMaxItems  = 1;
bool        g_EnableRAGReplyContext = true;
uint32_t    g_RAGReplyContextItems  = 1;

class OllamaRAGSystem;
OllamaRAGSystem* g_RAGSystem = nullptr;

// --------------------------------------------
// Blacklist: Prefixes for Commands (not chat)
// --------------------------------------------
std::vector<std::string> g_BlacklistCommands = {
    ".playerbots",
    "playerbot",
};

// --------------------------------------------
// Environment/Contextual Random Chatter Templates
// --------------------------------------------
std::vector<std::string> g_EnvCommentCreature;
std::vector<std::string> g_EnvCommentGameObject;
std::vector<std::string> g_EnvCommentEquippedItem;
std::vector<std::string> g_EnvCommentBagItem;
std::vector<std::string> g_EnvCommentBagItemSell;
std::vector<std::string> g_EnvCommentSpell;
std::vector<std::string> g_EnvCommentQuestArea;
std::vector<std::string> g_EnvCommentVendor;
std::vector<std::string> g_EnvCommentQuestgiver;
std::vector<std::string> g_EnvCommentBagSlots;
std::vector<std::string> g_EnvCommentDungeon;
std::vector<std::string> g_EnvCommentUnfinishedQuest;

// --------------------------------------------
// Guild-Specific Random Chatter Templates
// --------------------------------------------
std::vector<std::string> g_GuildEnvCommentGuildMember;
std::vector<std::string> g_GuildEnvCommentGuildRank;
std::vector<std::string> g_GuildEnvCommentGuildBank;
std::vector<std::string> g_GuildEnvCommentGuildMOTD;
std::vector<std::string> g_GuildEnvCommentGuildInfo;
std::vector<std::string> g_GuildEnvCommentGuildOnlineMembers;
std::vector<std::string> g_GuildEnvCommentGuildRaid;
std::vector<std::string> g_GuildEnvCommentGuildEndgame;
std::vector<std::string> g_GuildEnvCommentGuildStrategy;
std::vector<std::string> g_GuildEnvCommentGuildGroup;
std::vector<std::string> g_GuildEnvCommentGuildPvP;
std::vector<std::string> g_GuildEnvCommentGuildCommunity;

// --------------------------------------------
// Guild-Specific Random Chatter Configuration
// --------------------------------------------
bool        g_EnableGuildEventChatter             = true;
bool        g_EnableGuildRandomAmbientChatter      = true;
uint32_t    g_GuildRandomChatterChance             = 10;
uint32_t    g_GuildChatterBotCommentChance          = 25;
uint32_t    g_GuildChatterMaxBotsPerEvent           = 2;

// --------------------------------------------
// Guild-Specific Event Chatter Templates
// --------------------------------------------
std::string g_GuildEventTypeLevelUp = "";
std::string g_GuildEventTypeDungeonComplete = "";
std::string g_GuildEventTypeEpicGear = "";
std::string g_GuildEventTypeRareGear = "";
std::string g_GuildEventTypeGuildJoin = "";
std::string g_GuildEventTypeGuildLeave = "";
std::string g_GuildEventTypeGuildPromotion = "";
std::string g_GuildEventTypeGuildDemotion = "";
std::string g_GuildEventTypeGuildLogin = "";
std::string g_GuildEventTypeGuildAchievement = "";

// additional event-chatter hook type strings and chances
std::string g_EventTypeEnteredZone;
std::string g_EventTypeKilledByCreature;
std::string g_EventTypeReputationRank;
std::string g_EventTypeResurrected;
std::string g_EventTypeEnteredCombat;
std::string g_EventTypeLeftCombat;

uint32_t g_PlayerEventChance_EnteredZone = 0;
uint32_t g_BotEventChance_EnteredZone = 0;
uint32_t g_PlayerEventChance_KilledByCreature = 0;
uint32_t g_BotEventChance_KilledByCreature = 0;
uint32_t g_PlayerEventChance_ReputationRank = 0;
uint32_t g_BotEventChance_ReputationRank = 0;
uint32_t g_PlayerEventChance_Resurrected = 0;
uint32_t g_BotEventChance_Resurrected = 0;
uint32_t g_PlayerEventChance_EnteredCombat = 0;
uint32_t g_BotEventChance_EnteredCombat = 0;
uint32_t g_PlayerEventChance_LeftCombat = 0;
uint32_t g_BotEventChance_LeftCombat = 0;

// --------------------------------------------
// Event Chatter Templates
// --------------------------------------------
std::string g_EventTypeDefeated;           // "defeated"
std::string g_EventTypeDefeatedPlayer;     // "defeated player"
std::string g_EventTypePetDefeated;        // "pet defeated"
std::string g_EventTypeGotItem;            // "got item"
std::string g_EventTypeDied;               // "died"
std::string g_EventTypeCompletedQuest;     // "completed quest"
std::string g_EventTypeLearnedSpell;       // "learned spell"
std::string g_EventTypeRequestedDuel;      // "requested to duel"
std::string g_EventTypeStartedDueling;     // "started dueling"
std::string g_EventTypeWonDuel;            // "won duel against"
std::string g_EventTypeLeveledUp;          // "leveled up"
std::string g_EventTypeAchievement;        // "earned achievement"
std::string g_EventTypeUsedObject;         // "used object"

// Chance variables for normal events
int g_EventTypeDefeated_Chance = 0;
int g_EventTypeDefeatedPlayer_Chance = 0;
int g_EventTypePetDefeated_Chance = 0;
int g_EventTypeGotItem_Chance = 0;
int g_EventTypeDied_Chance = 0;
int g_EventTypeCompletedQuest_Chance = 0;
int g_EventTypeLearnedSpell_Chance = 0;
int g_EventTypeRequestedDuel_Chance = 0;
int g_EventTypeStartedDueling_Chance = 0;
int g_EventTypeWonDuel_Chance = 0;
int g_EventTypeLeveledUp_Chance = 0;
int g_EventTypeAchievement_Chance = 0;
int g_EventTypeUsedObject_Chance = 0;

// Chance variables for guild events
int g_GuildEventTypeEpicGear_Chance = 0;
int g_GuildEventTypeRareGear_Chance = 0;
int g_GuildEventTypeGuildJoin_Chance = 0;
int g_GuildEventTypeGuildLogin_Chance = 0;
int g_GuildEventTypeGuildLeave_Chance = 0;
int g_GuildEventTypeGuildPromotion_Chance = 0;
int g_GuildEventTypeGuildDemotion_Chance = 0;
int g_GuildEventTypeGuildAchievement_Chance = 0;
int g_GuildEventTypeLevelUp_Chance = 0;
int g_GuildEventTypeDungeonComplete_Chance = 0;

// Event Cooldown
uint32_t g_EventCooldownTime = 10;

// --------------------------------------------
// Channel Disable Settings
// --------------------------------------------
bool g_DisableForCustomChannels = false;
bool g_DisableForSayYell = false;
bool g_DisableForGuild = false;
bool g_DisableForParty = false;

// --------------------------------------------
// Typing Simulation Settings
// --------------------------------------------
bool g_EnableTypingSimulation = false;
uint32_t g_TypingSimulationBaseDelay = 1000;     // 1000ms base delay
uint32_t g_TypingSimulationDelayPerChar = 250;   // 250ms per character (4 chars/sec)

bool        g_SpeechSplitEnable               = true;
uint32_t    g_SpeechSplitMaxLineLength        = 230;
bool        g_SpeechSplitDropTrailingFragment = true;
uint32_t    g_SpeechSplitLineDelayMs          = 700;

// --------------------------------------------
// Emote-Augmented Chat
// --------------------------------------------
bool        g_EmoteChatEnable = true;
std::string g_EmoteChatVocabularyRaw;
std::string g_EmoteChatInstructionTemplate;
bool        g_EmoteActionRouting = true;


static std::vector<std::string> SplitString(const std::string& str, char delim)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delim))
    {
        // Trim whitespace from token
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos)
            tokens.push_back(token.substr(start, end - start + 1));
    }
    return tokens;
}

static std::vector<std::string> ParsePipeList(const std::string& s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, '|'))
        if (!tok.empty())
            out.push_back(tok);
    return out;
}

// --------------------------------------------
// Emote-Augmented Chat helpers
// --------------------------------------------

// Curated visual oneshots that read well alongside speech.
static const std::unordered_map<std::string, uint32>& EmoteChatBuiltinMap()
{
    static const std::unordered_map<std::string, uint32> m = {
        {"wave",    EMOTE_ONESHOT_WAVE},    {"laugh",   EMOTE_ONESHOT_LAUGH},
        {"cheer",   EMOTE_ONESHOT_CHEER},   {"cry",     EMOTE_ONESHOT_CRY},
        {"point",   EMOTE_ONESHOT_POINT},   {"bow",     EMOTE_ONESHOT_BOW},
        {"flex",    EMOTE_ONESHOT_FLEX},    {"applaud", EMOTE_ONESHOT_APPLAUD},
        {"roar",    EMOTE_ONESHOT_ROAR},    {"no",      EMOTE_ONESHOT_NO},
        {"yes",     EMOTE_ONESHOT_YES},     {"shrug",   EMOTE_ONESHOT_QUESTION},
        {"salute",  EMOTE_ONESHOT_SALUTE},  {"kneel",   EMOTE_ONESHOT_KNEEL},
        {"beg",     EMOTE_ONESHOT_BEG},     {"talk",    EMOTE_ONESHOT_TALK},
        {"rude",    EMOTE_ONESHOT_RUDE}
    };
    return m;
}

static std::set<std::string> EmoteChatAllowed()
{
    std::set<std::string> allowed;
    if (g_EmoteChatVocabularyRaw.empty())
    {
        for (auto const& kv : EmoteChatBuiltinMap())
            allowed.insert(kv.first);
        return allowed;
    }
    for (std::string const& n : ParsePipeList(g_EmoteChatVocabularyRaw))
    {
        // Trim surrounding whitespace so "wave | laugh | cheer" parses like "wave|laugh|cheer".
        size_t s = n.find_first_not_of(" \t");
        if (s == std::string::npos)
            continue;
        size_t e = n.find_last_not_of(" \t");
        std::string low = n.substr(s, e - s + 1);
        std::transform(low.begin(), low.end(), low.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (EmoteChatBuiltinMap().count(low))
            allowed.insert(low);
    }
    return allowed;
}

// Inflections / synonyms -> canonical curated emote name (single words only).
static std::map<std::string, std::string> const& EmoteInflectionMap()
{
    static const std::map<std::string, std::string> m = {
        {"waving","wave"},{"waved","wave"},{"waves","wave"},
        {"laughing","laugh"},{"laughs","laugh"},{"laughed","laugh"},
        {"cheering","cheer"},{"cheers","cheer"},{"cheered","cheer"},
        {"crying","cry"},{"cries","cry"},{"cried","cry"},
        {"pointing","point"},{"points","point"},{"pointed","point"},
        {"bowing","bow"},{"bows","bow"},{"bowed","bow"},
        {"flexing","flex"},{"flexes","flex"},{"flexed","flex"},
        {"applauding","applaud"},{"applauds","applaud"},{"applauded","applaud"},
        {"clapping","applaud"},{"claps","applaud"},{"clap","applaud"},
        {"roaring","roar"},{"roars","roar"},{"roared","roar"},
        {"nodding","yes"},{"nods","yes"},{"nod","yes"},{"agreeing","yes"},{"agrees","yes"},{"agree","yes"},
        {"disagreeing","no"},{"disagrees","no"},{"disagree","no"},
        {"shrugging","shrug"},{"shrugs","shrug"},{"shrugged","shrug"},
        {"saluting","salute"},{"salutes","salute"},{"saluted","salute"},
        {"kneeling","kneel"},{"kneels","kneel"},{"knelt","kneel"},
        {"begging","beg"},{"begs","beg"},{"begged","beg"},
        {"talking","talk"},{"talks","talk"},{"talked","talk"}
    };
    return m;
}

// Resolve a single lowercased tag name to an allowed curated emote id, or 0 if none.
// Tries exact, then an inflection/synonym; respects the active vocabulary (EmoteChatAllowed()).
static uint32_t FuzzyResolveEmote(const std::string& lowerName)
{
    std::string canon = lowerName;
    auto const& bm = EmoteChatBuiltinMap();
    if (!bm.count(canon))
    {
        auto inf = EmoteInflectionMap().find(lowerName);
        if (inf == EmoteInflectionMap().end())
            return 0;
        canon = inf->second;
    }
    auto it = bm.find(canon);
    if (it == bm.end())
        return 0;
    std::set<std::string> allowed = EmoteChatAllowed();
    if (!allowed.count(canon))
        return 0;
    return (uint32_t)it->second;
}

ChatLinePlan PlanChatLine(const std::string& line, bool proximity)
{
    ChatLinePlan plan;

    size_t startPos = line.find_first_not_of(" \t");
    if (startPos == std::string::npos || line[startPos] != '[')
    {
        plan.kind = ChatLinePlan::Speak; plan.text = line; return plan;
    }
    size_t close = line.find(']', startPos);
    if (close == std::string::npos)
    {
        plan.kind = ChatLinePlan::Speak; plan.text = line; return plan;
    }

    std::string inside = line.substr(startPos + 1, close - startPos - 1);
    size_t after = close + 1;
    while (after < line.size() && (line[after] == ' ' || line[after] == '\t'))
        ++after;
    std::string rest = (after < line.size()) ? line.substr(after) : "";

    std::string lname = SpeechTrim(inside);
    std::transform(lname.begin(), lname.end(), lname.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    bool singleWord = !lname.empty() && lname.find(' ') == std::string::npos;

    if (g_EmoteChatEnable && singleWord)
    {
        uint32_t id = FuzzyResolveEmote(lname);
        if (id != 0)
        {
            if (rest.empty()) { plan.kind = ChatLinePlan::Perform; plan.emote = id; }
            else { plan.kind = ChatLinePlan::PerformAndSpeak; plan.emote = id; plan.text = rest; }
            return plan;
        }
    }
    if (!g_EmoteChatEnable && singleWord && EmoteChatBuiltinMap().count(lname))
    {
        if (rest.empty()) { plan.kind = ChatLinePlan::Perform; plan.emote = 0; }
        else { plan.kind = ChatLinePlan::Speak; plan.text = rest; }
        return plan;
    }

    if (!g_EmoteActionRouting)
    {
        plan.kind = ChatLinePlan::Speak; plan.text = line; return plan;
    }

    std::string action = SpeechTrim(inside);
    if (action.empty())
    {
        plan.kind = ChatLinePlan::Speak; plan.text = rest; return plan;
    }

    if (proximity)
    {
        plan.kind = ChatLinePlan::ActionEmote;
        plan.text = action;
        plan.tail = rest;
    }
    else
    {
        plan.kind = ChatLinePlan::ActionText;
        plan.text = "*" + action + "*" + (rest.empty() ? "" : " " + rest);
    }
    return plan;
}

bool ApplyChatEmote(Player* bot, std::string& text)
{
    size_t start = text.find_first_not_of(" \t\n");
    if (start == std::string::npos || text[start] != '[')
        return false;
    size_t close = text.find(']', start);
    if (close == std::string::npos)
        return false;
    std::string name = text.substr(start + 1, close - start - 1);
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    bool singleWord = !name.empty() && name.find(' ') == std::string::npos;

    auto stripLeading = [&]()
    {
        size_t after = close + 1;
        while (after < text.size() && (text[after] == ' ' || text[after] == '\t'))
            ++after;
        text.erase(0, after);
    };

    if (!g_EmoteChatEnable)
    {
        if (singleWord && EmoteChatBuiltinMap().count(name))   // defensive: never speak a stray tag
            stripLeading();
        return false;
    }

    auto it = EmoteChatBuiltinMap().find(name);
    if (it == EmoteChatBuiltinMap().end())
        return false;   // truly unknown tag -> leave text intact (might be real content)
    std::set<std::string> allowed = EmoteChatAllowed();
    if (!allowed.count(name))
    {
        // Recognized built-in but filtered out of the active vocabulary: strip it so a
        // stray "[wave]" is never spoken literally, but perform nothing.
        if (singleWord)
            stripLeading();
        return false;
    }

    if (bot)
        bot->HandleEmoteCommand(it->second);
    stripLeading();
    return true;
}

std::string BuildEmoteChatInstruction()
{
    if (!g_EmoteChatEnable)
        return "";
    std::set<std::string> allowed = EmoteChatAllowed();
    if (allowed.empty())
        return "";
    std::string list;
    for (std::string const& n : allowed)
        list += (list.empty() ? "[" : " [") + n + "]";
    // Substitute the single {emote_list} placeholder manually (avoids pulling in
    // mod-ollama-chat-utilities.h, whose inline SplitString collides with this file's static one).
    std::string instr = g_EmoteChatInstructionTemplate;
    std::string const token = "{emote_list}";
    if (size_t pos = instr.find(token); pos != std::string::npos)
        instr.replace(pos, token.length(), list);
    return instr.empty() ? "" : " " + instr;
}

// ============================================================================
// Speech delivery: split a raw LLM reply into ordered, chat-safe lines.
// ============================================================================

static std::string SpeechTrim(const std::string& s)
{
    size_t b = s.find_first_not_of(" \t");
    if (b == std::string::npos)
        return "";
    size_t e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
}

static bool SpeechLooksComplete(const std::string& line)
{
    std::string s = SpeechTrim(line);
    if (s.empty())
        return false;
    char last = s.back();
    if (last == ']' || last == ')')
        return true;
    while (!s.empty() && (s.back() == '"' || s.back() == '\'' || s.back() == ' '))
        s.pop_back();
    if (s.empty())
        return false;
    last = s.back();
    if (last == '.' || last == '!' || last == '?' || last == ':' || last == '~')
        return true;
    if (s.size() >= 3 &&
        (unsigned char)s[s.size() - 3] == 0xE2 &&
        (unsigned char)s[s.size() - 2] == 0x80 &&
        (unsigned char)s[s.size() - 1] == 0xA6)
        return true;
    return false;
}

static void SpeechAppendLengthSplit(std::vector<std::string>& out,
                                    const std::string& seg, size_t maxLen)
{
    std::string s = seg;
    while (s.size() > maxLen)
    {
        size_t cut = std::string::npos;
        for (char t : {'.', '!', '?'})
        {
            size_t p = s.rfind(t, maxLen);
            if (p != std::string::npos && (cut == std::string::npos || p > cut))
                cut = p;
        }
        size_t cutEnd;
        if (cut != std::string::npos && cut + 1 >= maxLen / 2)
            cutEnd = cut + 1;
        else
        {
            size_t sp = s.rfind(' ', maxLen);
            cutEnd = (sp != std::string::npos && sp > 0) ? sp : maxLen;
        }
        std::string piece = SpeechTrim(s.substr(0, cutEnd));
        if (!piece.empty())
            out.push_back(piece);
        s = SpeechTrim(s.substr(cutEnd));
    }
    if (!s.empty())
        out.push_back(s);
}

std::vector<std::string> SplitChatResponse(const std::string& text)
{
    if (!g_SpeechSplitEnable)
    {
        std::vector<std::string> one;
        std::string t = SpeechTrim(text);
        if (!t.empty())
            one.push_back(t);
        return one;
    }

    std::string norm;
    norm.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i)
    {
        char c = text[i];
        if (c == '\\' && i + 1 < text.size() && (text[i + 1] == 'n' || text[i + 1] == 'r'))
        {
            norm.push_back('\n');
            ++i;
        }
        else if (c == '\r')
        {
            norm.push_back('\n');
        }
        else
        {
            norm.push_back(c);
        }
    }

    std::vector<std::string> lines;
    std::stringstream ss(norm);
    std::string seg;
    while (std::getline(ss, seg, '\n'))
    {
        std::string t = SpeechTrim(seg);
        if (t.empty())
            continue;
        if (g_SpeechSplitMaxLineLength > 0 && t.size() > g_SpeechSplitMaxLineLength)
            SpeechAppendLengthSplit(lines, t, g_SpeechSplitMaxLineLength);
        else
            lines.push_back(t);
    }

    if (g_SpeechSplitDropTrailingFragment && lines.size() > 1 &&
        !SpeechLooksComplete(lines.back()))
        lines.pop_back();

    return lines;
}

static void SpeechSplitSelfTest()
{
    struct Case { const char* name; std::string in; std::vector<std::string> out; };
    const std::vector<Case> cases = {
        { "plain",          "Hello there.",                         { "Hello there." } },
        { "two-lines",      "Line one.\nLine two.",                 { "Line one.", "Line two." } },
        { "double-newline", "Para one.\n\nPara two.",               { "Para one.", "Para two." } },
        { "literal-esc",    "Literal break.\\nSecond part.",        { "Literal break.", "Second part." } },
        { "frag-dropped",   "First thought.\nOh man, I'm",          { "First thought." } },
        { "bracket-kept",   "Looks good.\n[grins broadly]",         { "Looks good.", "[grins broadly]" } },
        { "single-frag",    "Oh man, I'm",                          { "Oh man, I'm" } },
    };

    int passed = 0;
    for (const Case& c : cases)
    {
        std::vector<std::string> got = SplitChatResponse(c.in);
        bool ok = (got == c.out);
        if (ok)
            ++passed;
        else
        {
            std::string g;
            for (const std::string& l : got) { g += "[" + l + "]"; }
            LOG_ERROR("server.loading",
                      "[Ollama Chat] SpeechSplit self-test FAIL ({}) -> {}", c.name, g);
        }
    }
    LOG_INFO("server.loading",
             "[Ollama Chat] SpeechSplit self-test: {}/{} passed", passed, (int)cases.size());
}

std::string EmitBotLines(const std::string& response,
                         const std::function<void(const std::string&)>& sendLine)
{
    std::vector<std::string> lines = SplitChatResponse(response);
    if (lines.empty())
        return "";

    std::string joined;
    for (size_t i = 0; i < lines.size(); ++i)
    {
        if (i > 0 && g_SpeechSplitLineDelayMs > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(g_SpeechSplitLineDelayMs));
        sendLine(lines[i]);
        if (i > 0)
            joined += ' ';
        joined += lines[i];
    }
    return joined;
}

// Load Bot Personalities from Database
void LoadBotPersonalityList()
{    
    // Let's make sure our user has sourced the required sql file to add the new table
    QueryResult tableExists = CharacterDatabase.Query("SELECT * FROM information_schema.tables WHERE table_schema = 'acore_characters' AND table_name = 'mod_ollama_chat_personality' LIMIT 1");
    if (!tableExists)
    {
        LOG_ERROR("server.loading", "[Ollama Chat] Please source the required database table first");
        return;
    }

    QueryResult result = CharacterDatabase.Query("SELECT guid,personality FROM mod_ollama_chat_personality");

    if (!result)
    {
        return;
    }
    if (result->GetRowCount() == 0)
    {
        return;
    }    

    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[Ollama Chat] Fetching Bot Personality List into array");
    }

    do
    {
        uint64_t personalityBotGUID = result->Fetch()[0].Get<uint64_t>();
        std::string personalityKey = result->Fetch()[1].Get<std::string>();
        g_BotPersonalityList[personalityBotGUID] = personalityKey;
    } while (result->NextRow());
}

std::string GetMultiLineConfigValue(const std::string& configFilePath, const std::string& key)
{
    std::ifstream infile(configFilePath);
    if (!infile) return "";

    std::string line;
    std::string value;
    bool foundKey = false;
    while (std::getline(infile, line))
    {
        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
        if (trimmed.empty() || trimmed[0] == '#')
            continue;
        size_t pos = trimmed.find('=');
        if (!foundKey && pos != std::string::npos) {
            std::string possibleKey = trimmed.substr(0, pos);
            possibleKey.erase(possibleKey.find_last_not_of(" \t\r\n") + 1);
            if (possibleKey == key) {
                foundKey = true;
                std::string afterEq = trimmed.substr(pos + 1);
                afterEq.erase(0, afterEq.find_first_not_of(" \t\r\n"));
                value += afterEq;
                continue;
            }
        }
        else if (foundKey) {
            // New config key or section
            if (trimmed.find('=') != std::string::npos && trimmed.find('[') == std::string::npos)
                break;
            if (!value.empty()) value += "\n";
            value += trimmed;
        }
    }

    return value;
}

void LoadOllamaChatConfig()
{
    g_SayDistance                     = sConfigMgr->GetOption<float>("OllamaChat.SayDistance", 30.0f);
    g_YellDistance                    = sConfigMgr->GetOption<float>("OllamaChat.YellDistance", 100.0f);
    
    // Load per-channel-type reply chances
    g_PlayerReplyChance_Say           = sConfigMgr->GetOption<uint32_t>("OllamaChat.PlayerReplyChance.Say", 90);
    g_BotReplyChance_Say              = sConfigMgr->GetOption<uint32_t>("OllamaChat.BotReplyChance.Say", 10);
    g_PlayerReplyChance_Channel       = sConfigMgr->GetOption<uint32_t>("OllamaChat.PlayerReplyChance.Channel", 50);
    g_BotReplyChance_Channel          = sConfigMgr->GetOption<uint32_t>("OllamaChat.BotReplyChance.Channel", 5);
    g_PlayerReplyChance_Party         = sConfigMgr->GetOption<uint32_t>("OllamaChat.PlayerReplyChance.Party", 90);
    g_BotReplyChance_Party            = sConfigMgr->GetOption<uint32_t>("OllamaChat.BotReplyChance.Party", 10);
    g_PlayerReplyChance_Guild         = sConfigMgr->GetOption<uint32_t>("OllamaChat.PlayerReplyChance.Guild", 70);
    g_BotReplyChance_Guild            = sConfigMgr->GetOption<uint32_t>("OllamaChat.BotReplyChance.Guild", 5);
    
    g_MaxBotsToPick                   = sConfigMgr->GetOption<uint32_t>("OllamaChat.MaxBotsToPick", 2);
    g_OllamaUrl                       = sConfigMgr->GetOption<std::string>("OllamaChat.Url", "http://localhost:11434/api/generate");
    g_OllamaModel                     = sConfigMgr->GetOption<std::string>("OllamaChat.Model", "llama3.2:1b");
    g_OllamaNumPredict                = sConfigMgr->GetOption<uint32_t>("OllamaChat.NumPredict", 40);
    g_SoftStopEnable                  = sConfigMgr->GetOption<bool>("OllamaChat.SoftStopEnable", true);
    g_OllamaTemperature               = sConfigMgr->GetOption<float>("OllamaChat.Temperature", 0.8f);
    g_OllamaTopP                      = sConfigMgr->GetOption<float>("OllamaChat.TopP", 0.95f);
    g_OllamaRepeatPenalty             = sConfigMgr->GetOption<float>("OllamaChat.RepeatPenalty", 1.1f);
    g_OllamaTopK                      = sConfigMgr->GetOption<uint32_t>("OllamaChat.TopK", 0);
    g_OllamaMinP                      = sConfigMgr->GetOption<float>("OllamaChat.MinP", 0.0f);
    g_OllamaPresencePenalty           = sConfigMgr->GetOption<float>("OllamaChat.PresencePenalty", 0.0f);
    g_OllamaFrequencyPenalty          = sConfigMgr->GetOption<float>("OllamaChat.FrequencyPenalty", 0.0f);
    g_OllamaNumCtx                    = sConfigMgr->GetOption<uint32_t>("OllamaChat.NumCtx", 0);
    g_OllamaNumThreads                = sConfigMgr->GetOption<uint32_t>("OllamaChat.NumThreads", 0);
    g_OllamaStop                      = sConfigMgr->GetOption<std::string>("OllamaChat.Stop", "");
    g_OllamaSystemPrompt              = sConfigMgr->GetOption<std::string>("OllamaChat.SystemPrompt", "");
    g_OllamaSeed                      = sConfigMgr->GetOption<std::string>("OllamaChat.Seed", "");

    g_MaxConcurrentQueries            = sConfigMgr->GetOption<uint32_t>("OllamaChat.MaxConcurrentQueries", 0);

    g_Enable                          = sConfigMgr->GetOption<bool>("OllamaChat.Enable", true);
    g_DisableRepliesInCombat          = sConfigMgr->GetOption<bool>("OllamaChat.DisableRepliesInCombat", true);
    g_EnableRandomChatter             = sConfigMgr->GetOption<bool>("OllamaChat.EnableRandomChatter", true);
    g_EnableEventChatter              = sConfigMgr->GetOption<bool>("OllamaChat.EnableEventChatter", true);
    g_EnableWhisperReplies            = sConfigMgr->GetOption<bool>("OllamaChat.EnableWhisperReplies", false);

    g_DebugEnabled                    = sConfigMgr->GetOption<bool>("OllamaChat.DebugEnabled", false);
    g_DebugShowFullPrompt             = sConfigMgr->GetOption<bool>("OllamaChat.DebugShowFullPrompt", false);

    g_MinRandomInterval               = sConfigMgr->GetOption<uint32_t>("OllamaChat.MinRandomInterval", 45);
    g_MaxRandomInterval               = sConfigMgr->GetOption<uint32_t>("OllamaChat.MaxRandomInterval", 180);
    g_RandomChatterRealPlayerDistance = sConfigMgr->GetOption<float>("OllamaChat.RandomChatterRealPlayerDistance", 40.0f);
    g_RandomChatterBotCommentChance   = sConfigMgr->GetOption<uint32_t>("OllamaChat.RandomChatterBotCommentChance", 25);
    g_RandomChatterMaxBotsPerPlayer   = sConfigMgr->GetOption<uint32_t>("OllamaChat.RandomChatterMaxBotsPerPlayer", 2);

    g_EnableGuildRandomAmbientChatter = sConfigMgr->GetOption<bool>("OllamaChat.EnableGuildRandomAmbientChatter", true);
    g_GuildRandomChatterChance        = sConfigMgr->GetOption<uint32_t>("OllamaChat.GuildRandomChatterChance", 10);

    g_EventChatterRealPlayerDistance = sConfigMgr->GetOption<float>("OllamaChat.EventChatterRealPlayerDistance", 40.0f);
    g_EventChatterBotCommentChance   = sConfigMgr->GetOption<uint32_t>("OllamaChat.EventChatterBotCommentChance", 15);
    g_EventChatterBotSelfCommentChance = sConfigMgr->GetOption<uint32_t>("OllamaChat.EventChatterBotSelfCommentChance", 5);
    g_EventChatterMaxBotsPerPlayer   = sConfigMgr->GetOption<uint32_t>("OllamaChat.EventChatterMaxBotsPerPlayer", 2);

    g_EnableRPPersonalities           = sConfigMgr->GetOption<bool>("OllamaChat.EnableRPPersonalities", false);

    g_RandomChatterPromptTemplate     = sConfigMgr->GetOption<std::string>("OllamaChat.RandomChatterPromptTemplate", "");

    // Load random chatter prompt variations
    std::string variationsStr = sConfigMgr->GetOption<std::string>("OllamaChat.RandomChatterPromptVariations", "");
    g_RandomChatterPromptVariations.clear();
    if (!variationsStr.empty())
    {
        std::stringstream ss(variationsStr);
        std::string variation;
        while (std::getline(ss, variation, '|'))
        {
            if (!variation.empty())
            {
                g_RandomChatterPromptVariations.push_back(variation);
            }
        }
    }

    // Load random chatter question variations
    std::string questionsStr = sConfigMgr->GetOption<std::string>("OllamaChat.RandomChatterQuestionVariations", "");
    g_RandomChatterQuestionVariations.clear();
    if (!questionsStr.empty())
    {
        std::stringstream ss(questionsStr);
        std::string question;
        while (std::getline(ss, question, '|'))
        {
            if (!question.empty())
            {
                g_RandomChatterQuestionVariations.push_back(question);
            }
        }
    }

    g_EventChatterPromptTemplate     = sConfigMgr->GetOption<std::string>("OllamaChat.EventChatterPromptTemplate", "");

    g_ChatPromptTemplate              = sConfigMgr->GetOption<std::string>("OllamaChat.ChatPromptTemplate", "");
    
    g_ChatExtraInfoTemplate           = sConfigMgr->GetOption<std::string>("OllamaChat.ChatExtraInfoTemplate", "");

    g_DefaultPersonalityPrompt        = sConfigMgr->GetOption<std::string>("OllamaChat.DefaultPersonalityPrompt", "");

    g_MaxConversationHistory          = sConfigMgr->GetOption<uint32_t>("OllamaChat.MaxConversationHistory", 5);
    g_ConversationHistorySaveInterval = sConfigMgr->GetOption<uint32_t>("OllamaChat.ConversationHistorySaveInterval", 10);

    g_ChatHistoryHeaderTemplate       = sConfigMgr->GetOption<std::string>("OllamaChat.ChatHistoryHeaderTemplate", "");
    g_ChatHistoryLineTemplate         = sConfigMgr->GetOption<std::string>("OllamaChat.ChatHistoryLineTemplate", "");
    g_ChatHistoryFooterTemplate       = sConfigMgr->GetOption<std::string>("OllamaChat.ChatHistoryFooterTemplate", "");

    g_EnableChatBotSnapshotTemplate   = sConfigMgr->GetOption<bool>("OllamaChat.EnableChatBotSnapshotTemplate", false);
    g_ChatBotSnapshotTemplate         = sConfigMgr->GetOption<std::string>("OllamaChat.ChatBotSnapshotTemplate", "");
    g_SnapshotMaxQuests    = sConfigMgr->GetOption<uint32_t>("OllamaChat.SnapshotMaxQuests", 3);
    g_SnapshotMaxGroup     = sConfigMgr->GetOption<uint32_t>("OllamaChat.SnapshotMaxGroup", 4);
    g_SnapshotMaxCreatures = sConfigMgr->GetOption<uint32_t>("OllamaChat.SnapshotMaxCreatures", 2);
    g_SnapshotMaxPlayers   = sConfigMgr->GetOption<uint32_t>("OllamaChat.SnapshotMaxPlayers", 2);
    g_SnapshotMaxSpells    = sConfigMgr->GetOption<uint32_t>("OllamaChat.SnapshotMaxSpells", 2);

    g_EnableAntiRepetition   = sConfigMgr->GetOption<bool>("OllamaChat.AntiRepetition.Enable", true);
    g_AntiRepetitionWindow   = sConfigMgr->GetOption<uint32_t>("OllamaChat.AntiRepetition.Window", 6);
    g_EnableCrossBotAntiRepetition   = sConfigMgr->GetOption<bool>("OllamaChat.CrossBotAntiRepetition.Enable", false);
    g_CrossBotAntiRepetitionWindow   = sConfigMgr->GetOption<uint32_t>("OllamaChat.CrossBotAntiRepetition.Window", 2);
    g_CrossBotAntiRepetitionMaxLines = sConfigMgr->GetOption<uint32_t>("OllamaChat.CrossBotAntiRepetition.MaxLines", 6);
    g_CrossBotAntiRepetitionTemplate = sConfigMgr->GetOption<std::string>("OllamaChat.CrossBotAntiRepetitionTemplate", " [Others near you recently said these - say something different, do not echo them: {nearby_replies}]");
    g_AntiRepetitionTemplate = sConfigMgr->GetOption<std::string>("OllamaChat.AntiRepetitionTemplate",
        " [You recently said these - do not repeat them; say something clearly different and fresh: {recent_replies}]");

    g_EnableConversationThreading   = sConfigMgr->GetOption<bool>("OllamaChat.ConversationThreading.Enable", true);
    g_ConversationThreadWindow      = sConfigMgr->GetOption<uint32_t>("OllamaChat.ConversationThreadWindow", 8);
    g_ConversationThreadTTLMinutes  = sConfigMgr->GetOption<uint32_t>("OllamaChat.ConversationThreadTTLMinutes", 30);
    g_ConversationThreadMaxChannels = sConfigMgr->GetOption<uint32_t>("OllamaChat.ConversationThreadMaxChannels", 64);
    g_ConversationThreadTemplate    = sConfigMgr->GetOption<std::string>("OllamaChat.ConversationThreadTemplate",
        " [Recent talk in this channel - join in, reply to the latest naturally, do not just repeat others: {thread}]");

    g_EnableChannelFrames = sConfigMgr->GetOption<bool>("OllamaChat.ChannelFrames.Enable", true);
    g_EnableChannelTopics = sConfigMgr->GetOption<bool>("OllamaChat.ChannelTopics.Enable", true);

    static const char* kFrameKeys[8] = {
        "OllamaChat.ChannelFrame.Guild", "OllamaChat.ChannelFrame.Party",
        "OllamaChat.ChannelFrame.Raid",  "OllamaChat.ChannelFrame.Say",
        "OllamaChat.ChannelFrame.Yell",  "OllamaChat.ChannelFrame.General",
        "OllamaChat.ChannelFrame.Trade", "OllamaChat.ChannelFrame.Others"
    };
    static const char* kFrameDef[8] = {
        "You're chatting in your guild channel, among guildmates who know you well - guild life, shared plans, camaraderie.",
        "You're speaking to your adventuring group - the task at hand, your companions, what's right in front of you.",
        "You're addressing the whole raid - the fight ahead, coordination, the stakes of a big undertaking.",
        "You speak aloud to whoever stands nearby - remark on your surroundings, passers-by, the moment.",
        "You raise your voice for all nearby to hear - a call, a warning, an announcement.",
        "You're posting in the regional General channel - anyone in the area might read it - idle musings, questions, regional happenings.",
        "You're posting in the Trade channel - goods, services, deals, and the usual rabble of a marketplace.",
        "You're speaking in a public channel to a broad audience."
    };
    for (int i = 0; i < 8; ++i)
        g_ChannelFrames[i] = sConfigMgr->GetOption<std::string>(kFrameKeys[i], kFrameDef[i]);

    static const char* kTopicKeys[8] = {
        "OllamaChat.ChannelTopics.Guild", "OllamaChat.ChannelTopics.Party",
        "OllamaChat.ChannelTopics.Raid",  "OllamaChat.ChannelTopics.Say",
        "OllamaChat.ChannelTopics.Yell",  "OllamaChat.ChannelTopics.General",
        "OllamaChat.ChannelTopics.Trade", "OllamaChat.ChannelTopics.Others"
    };
    static const char* kTopicDef[8] = {
        "a guild raid being planned|a guildmate's recent deed|recruiting new blood|the state of the guild bank|an upcoming gathering|an old shared story",
        "the next pull ahead|loot that just dropped|a narrow escape|where to head next|a companion's gear or skill",
        "the great foe ahead|holding the line together|who stands where|the spoils of victory|steeling nerves before the fight",
        "the scenery or weather|a passing stranger|a beast prowling nearby|news of this town|a rumor lately overheard",
        "a call for aid|warning of danger near|seeking companions for a venture|a triumphant boast",
        "idle musings|a question for the region|something seen on the road|local happenings|seeking advice",
        "",
        ""
    };
    for (int i = 0; i < 8; ++i)
        g_ChannelTopics[i] = ParsePipeList(sConfigMgr->GetOption<std::string>(kTopicKeys[i], kTopicDef[i]));

    g_ActivityChatterEnable = sConfigMgr->GetOption<bool>("OllamaChat.ActivityChatter.Enable", true);
    g_ActivityChatterChance = sConfigMgr->GetOption<uint32_t>("OllamaChat.ActivityChatter.Chance", 40);

    g_ActivityTopicsSocial  = ParsePipeList(sConfigMgr->GetOption<std::string>("OllamaChat.ActivityTopics.Social",
        "the good company at this gathering|how pleasant it is to relax with others for a while|an old friend you ran into here"));
    g_ActivityTopicsFishing = ParsePipeList(sConfigMgr->GetOption<std::string>("OllamaChat.ActivityTopics.Fishing",
        "the fish finally biting|the big catch you are hoping to land|how peaceful it is to fish by the water|the one that got away"));
    g_ActivityTopicsGather  = ParsePipeList(sConfigMgr->GetOption<std::string>("OllamaChat.ActivityTopics.Gather",
        "a rich vein or herb you just spotted|the materials you have been collecting|how the day's gathering is going"));
    g_ActivityTopicsCraft   = ParsePipeList(sConfigMgr->GetOption<std::string>("OllamaChat.ActivityTopics.Craft",
        "the piece you are working on|honing your trade skill|a recipe you have been wanting to try|the quality of your latest work"));
    g_ActivityTopicsDuel    = ParsePipeList(sConfigMgr->GetOption<std::string>("OllamaChat.ActivityTopics.Duel",
        "a friendly duel to test your skills|how the sparring is going|a worthy opponent you just faced"));

    static const char* kWeightKeys[8] = {
        "OllamaChat.ChannelWeight.Guild", "OllamaChat.ChannelWeight.Party",
        "OllamaChat.ChannelWeight.Raid",  "OllamaChat.ChannelWeight.Say",
        "OllamaChat.ChannelWeight.Yell",  "OllamaChat.ChannelWeight.General",
        "OllamaChat.ChannelWeight.Trade", "OllamaChat.ChannelWeight.Others"
    };
    static const uint32_t kWeightDef[8] = { 25, 20, 15, 30, 5, 15, 0, 0 };
    for (int i = 0; i < 8; ++i)
        g_ChannelWeights[i] = sConfigMgr->GetOption<uint32_t>(kWeightKeys[i], kWeightDef[i]);

    g_EnableChatHistory               = sConfigMgr->GetOption<bool>("OllamaChat.EnableChatHistory", true);

    // Bot-Player Sentiment Tracking
    g_EnableSentimentTracking         = sConfigMgr->GetOption<bool>("OllamaChat.EnableSentimentTracking", true);
    g_SentimentDefaultValue           = sConfigMgr->GetOption<float>("OllamaChat.SentimentDefaultValue", 0.5f);
    g_SentimentAdjustmentStrength     = sConfigMgr->GetOption<float>("OllamaChat.SentimentAdjustmentStrength", 0.1f);
    g_SentimentSaveInterval           = sConfigMgr->GetOption<uint32_t>("OllamaChat.SentimentSaveInterval", 10);
    g_SentimentAnalysisPrompt         = sConfigMgr->GetOption<std::string>("OllamaChat.SentimentAnalysisPrompt", "Analyze the sentiment of this message: \"{message}\". Respond only with: POSITIVE, NEGATIVE, or NEUTRAL.");
    g_SentimentPromptTemplate         = sConfigMgr->GetOption<std::string>("OllamaChat.SentimentPromptTemplate", "Your relationship sentiment with {player_name} is {sentiment_value} (0.0=hostile, 0.5=neutral, 1.0=friendly). Use this to guide your tone and response.");

    // RAG (Retrieval-Augmented Generation) System
    g_EnableRAG                       = sConfigMgr->GetOption<bool>("OllamaChat.EnableRAG", false);
    g_RAGDataPath                     = sConfigMgr->GetOption<std::string>("OllamaChat.RAGDataPath", "rag/");
    g_RAGMaxRetrievedItems            = sConfigMgr->GetOption<uint32_t>("OllamaChat.RAGMaxRetrievedItems", 3);
    g_RAGSimilarityThreshold          = sConfigMgr->GetOption<float>("OllamaChat.RAGSimilarityThreshold", 0.3f);
    g_RAGPromptTemplate               = sConfigMgr->GetOption<std::string>("OllamaChat.RAGPromptTemplate", "RELEVANT INFORMATION:\n{rag_info}\nUse this information to provide accurate and detailed responses when applicable.");
    g_RAGExpandReferences             = sConfigMgr->GetOption<bool>("OllamaChat.RAGExpandReferences", true);
    g_RAGMaxReferences                = sConfigMgr->GetOption<uint32_t>("OllamaChat.RAGMaxReferences", 5);
    g_RAGRandomizeSelection           = sConfigMgr->GetOption<bool>("OllamaChat.RAGRandomizeSelection", true);
    g_RAGSelectionPoolSize            = sConfigMgr->GetOption<uint32_t>("OllamaChat.RAGSelectionPoolSize", 8);
    g_RAGImprovedScoring              = sConfigMgr->GetOption<bool>("OllamaChat.RAGImprovedScoring", true);
    g_EnableRAGInitiated              = sConfigMgr->GetOption<bool>("OllamaChat.EnableRAGInitiated", true);
    g_RAGInitiatedMaxItems            = sConfigMgr->GetOption<uint32_t>("OllamaChat.RAGInitiatedMaxItems", 1);
    g_EnableRAGReplyContext           = sConfigMgr->GetOption<bool>("OllamaChat.EnableRAGReplyContext", true);
    g_RAGReplyContextItems            = sConfigMgr->GetOption<uint32_t>("OllamaChat.RAGReplyContextItems", 1);

    g_ThinkModeEnableForModule        = sConfigMgr->GetOption<bool>("OllamaChat.ThinkModeEnableForModule", false);

    // Typing Simulation
    g_EnableTypingSimulation          = sConfigMgr->GetOption<bool>("OllamaChat.EnableTypingSimulation", false);
    g_TypingSimulationBaseDelay       = sConfigMgr->GetOption<uint32_t>("OllamaChat.TypingSimulationBaseDelay", 1000);
    g_TypingSimulationDelayPerChar    = sConfigMgr->GetOption<uint32_t>("OllamaChat.TypingSimulationDelayPerChar", 250);

    g_SpeechSplitEnable               = sConfigMgr->GetOption<bool>("OllamaChat.SpeechSplitEnable", true);
    g_SpeechSplitMaxLineLength        = sConfigMgr->GetOption<uint32_t>("OllamaChat.SpeechSplitMaxLineLength", 230);
    g_SpeechSplitDropTrailingFragment = sConfigMgr->GetOption<bool>("OllamaChat.SpeechSplitDropTrailingFragment", true);
    g_SpeechSplitLineDelayMs          = sConfigMgr->GetOption<uint32_t>("OllamaChat.SpeechSplitLineDelayMs", 700);

    // Emote-Augmented Chat
    g_EmoteChatEnable             = sConfigMgr->GetOption<bool>("OllamaChat.EmoteChat.Enable", true);
    g_EmoteChatVocabularyRaw      = sConfigMgr->GetOption<std::string>("OllamaChat.EmoteChat.Vocabulary", "");
    g_EmoteActionRouting = sConfigMgr->GetOption<bool>("OllamaChat.EmoteActionRouting", true);
    g_EmoteChatInstructionTemplate = sConfigMgr->GetOption<std::string>(
        "OllamaChat.EmoteChat.InstructionTemplate",
        "You may begin your reply with ONE emote in square brackets from this list, but only if it "
        "genuinely fits the mood; otherwise omit it entirely. Allowed: {emote_list}. Example: [cheer] We did it!");

    g_EventTypeDefeated           = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeDefeated", "");
    g_EventTypeDefeatedPlayer     = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeDefeatedPlayer", "");
    g_EventTypePetDefeated        = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypePetDefeated", "");
    g_EventTypeGotItem            = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeGotItem", "");
    g_EventTypeDied               = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeDied", "");
    g_EventTypeCompletedQuest     = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeCompletedQuest", "");
    g_EventTypeLearnedSpell       = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeLearnedSpell", "");
    g_EventTypeRequestedDuel      = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeRequestedDuel", "");
    g_EventTypeStartedDueling     = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeStartedDueling", "");
    g_EventTypeWonDuel            = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeWonDuel", "");
    g_EventTypeLeveledUp          = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeLeveledUp", "");
    g_EventTypeAchievement        = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeAchievement", "");
    g_EventTypeUsedObject         = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeUsedObject", "");

    // additional event-chatter hook type strings
    g_EventTypeEnteredZone      = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeEnteredZone", "entered zone");
    g_EventTypeKilledByCreature = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeKilledByCreature", "killed by creature");
    g_EventTypeReputationRank   = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeReputationRank", "reputation rank");
    g_EventTypeResurrected      = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeResurrected", "resurrected");
    g_EventTypeEnteredCombat    = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeEnteredCombat", "entered combat");
    g_EventTypeLeftCombat       = sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeLeftCombat", "left combat");

    // per-event player/bot chances (default 0 = off)
    g_PlayerEventChance_EnteredZone      = sConfigMgr->GetOption<uint32_t>("OllamaChat.PlayerEventChance.EnteredZone", 0);
    g_BotEventChance_EnteredZone         = sConfigMgr->GetOption<uint32_t>("OllamaChat.BotEventChance.EnteredZone", 0);
    g_PlayerEventChance_KilledByCreature = sConfigMgr->GetOption<uint32_t>("OllamaChat.PlayerEventChance.KilledByCreature", 0);
    g_BotEventChance_KilledByCreature    = sConfigMgr->GetOption<uint32_t>("OllamaChat.BotEventChance.KilledByCreature", 0);
    g_PlayerEventChance_ReputationRank   = sConfigMgr->GetOption<uint32_t>("OllamaChat.PlayerEventChance.ReputationRank", 0);
    g_BotEventChance_ReputationRank      = sConfigMgr->GetOption<uint32_t>("OllamaChat.BotEventChance.ReputationRank", 0);
    g_PlayerEventChance_Resurrected      = sConfigMgr->GetOption<uint32_t>("OllamaChat.PlayerEventChance.Resurrected", 0);
    g_BotEventChance_Resurrected         = sConfigMgr->GetOption<uint32_t>("OllamaChat.BotEventChance.Resurrected", 0);
    g_PlayerEventChance_EnteredCombat    = sConfigMgr->GetOption<uint32_t>("OllamaChat.PlayerEventChance.EnteredCombat", 0);
    g_BotEventChance_EnteredCombat       = sConfigMgr->GetOption<uint32_t>("OllamaChat.BotEventChance.EnteredCombat", 0);
    g_PlayerEventChance_LeftCombat       = sConfigMgr->GetOption<uint32_t>("OllamaChat.PlayerEventChance.LeftCombat", 0);
    g_BotEventChance_LeftCombat          = sConfigMgr->GetOption<uint32_t>("OllamaChat.BotEventChance.LeftCombat", 0);

    // Load extra blacklist commands from config (comma-separated list)
    std::string extraBlacklist = sConfigMgr->GetOption<std::string>("OllamaChat.BlacklistCommands", "");
    if (!extraBlacklist.empty())
    {
        std::vector<std::string> extraList = SplitString(extraBlacklist, ',');
        for (const auto& cmd : extraList)
        {
            g_BlacklistCommands.push_back(cmd);
        }
    }

    LoadPersonalityTemplatesFromDB();

    g_queryManager.setMaxConcurrentQueries(g_MaxConcurrentQueries);

    // Loads the environment random chatter message templates for each type.
    // Each config option is a pipe-separated list of string templates,
    // using {} as a placeholder for named substitutions.
    // Helper to load a multi-line config option into a std::vector<std::string>
    auto LoadEnvCommentVector = [](const char* key, const std::vector<std::string>& defaults = {}) -> std::vector<std::string>
    {
        std::string val = sConfigMgr->GetOption<std::string>(key, "");
        std::vector<std::string> result;
        std::istringstream iss(val);
        std::string token;
        while (std::getline(iss, token, '|')) { // Split by '|'
            // Trim whitespace from token
            size_t start = token.find_first_not_of(" \t\r\n");
            size_t end = token.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos)
                result.push_back(token.substr(start, end - start + 1));
        }
        if (result.empty() && !defaults.empty())
            return defaults;
        return result;
    };

    g_EnvCommentCreature        = LoadEnvCommentVector("OllamaChat.EnvCommentCreature", { "" });
    g_EnvCommentGameObject      = LoadEnvCommentVector("OllamaChat.EnvCommentGameObject", { "" });
    g_EnvCommentEquippedItem    = LoadEnvCommentVector("OllamaChat.EnvCommentEquippedItem", { "" });
    g_EnvCommentBagItem         = LoadEnvCommentVector("OllamaChat.EnvCommentBagItem", { "" });
    g_EnvCommentBagItemSell     = LoadEnvCommentVector("OllamaChat.EnvCommentBagItemSell", { "" });
    g_EnvCommentSpell           = LoadEnvCommentVector("OllamaChat.EnvCommentSpell", { "" });
    g_EnvCommentQuestArea       = LoadEnvCommentVector("OllamaChat.EnvCommentQuestArea", { "" });
    g_EnvCommentVendor          = LoadEnvCommentVector("OllamaChat.EnvCommentVendor", { "" });
    g_EnvCommentQuestgiver      = LoadEnvCommentVector("OllamaChat.EnvCommentQuestgiver", { "" });
    g_EnvCommentBagSlots        = LoadEnvCommentVector("OllamaChat.EnvCommentBagSlots", { "" });
    g_EnvCommentDungeon         = LoadEnvCommentVector("OllamaChat.EnvCommentDungeon", { "" });
    g_EnvCommentUnfinishedQuest = LoadEnvCommentVector("OllamaChat.EnvCommentUnfinishedQuest", { "" });

    // Guild-specific random chatter templates
    g_GuildEnvCommentGuildMember = LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildMember", { "" });
    g_GuildEnvCommentGuildRank = LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildRank", { "" });
    g_GuildEnvCommentGuildBank = LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildBank", { "" });
    g_GuildEnvCommentGuildMOTD = LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildMOTD", { "" });
    g_GuildEnvCommentGuildInfo = LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildInfo", { "" });
    g_GuildEnvCommentGuildOnlineMembers = LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildOnlineMembers", { "" });
    g_GuildEnvCommentGuildRaid = LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildRaid", { "" });
    g_GuildEnvCommentGuildEndgame = LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildEndgame", { "" });
    g_GuildEnvCommentGuildStrategy = LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildStrategy", { "" });
    g_GuildEnvCommentGuildGroup = LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildGroup", { "" });
    g_GuildEnvCommentGuildPvP = LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildPvP", { "" });
    g_GuildEnvCommentGuildCommunity = LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildCommunity", { "" });

    // Guild-specific configuration
    g_EnableGuildEventChatter = sConfigMgr->GetOption<bool>("OllamaChat.EnableGuildEventChatter", true);
    g_GuildChatterBotCommentChance = sConfigMgr->GetOption<uint32_t>("OllamaChat.GuildChatterBotCommentChance", 25);
    g_GuildChatterMaxBotsPerEvent = sConfigMgr->GetOption<uint32_t>("OllamaChat.GuildChatterMaxBotsPerEvent", 2);

    // Guild-specific event templates
    g_GuildEventTypeLevelUp = sConfigMgr->GetOption<std::string>("OllamaChat.GuildEventTypeLevelUp", "");
    g_GuildEventTypeDungeonComplete = sConfigMgr->GetOption<std::string>("OllamaChat.GuildEventTypeDungeonComplete", "");
    g_GuildEventTypeEpicGear = sConfigMgr->GetOption<std::string>("OllamaChat.GuildEventTypeEpicGear", "");
    g_GuildEventTypeRareGear = sConfigMgr->GetOption<std::string>("OllamaChat.GuildEventTypeRareGear", "");
    g_GuildEventTypeGuildJoin = sConfigMgr->GetOption<std::string>("OllamaChat.GuildEventTypeGuildJoin", "");
    g_GuildEventTypeGuildLogin = sConfigMgr->GetOption<std::string>("OllamaChat.GuildEventTypeGuildLogin", "");
    g_GuildEventTypeGuildLeave = sConfigMgr->GetOption<std::string>("OllamaChat.GuildEventTypeGuildLeave", "");
    g_GuildEventTypeGuildPromotion = sConfigMgr->GetOption<std::string>("OllamaChat.GuildEventTypeGuildPromotion", "");
    g_GuildEventTypeGuildDemotion = sConfigMgr->GetOption<std::string>("OllamaChat.GuildEventTypeGuildDemotion", "");

    // Load chance variables for normal events
    g_EventTypeDefeated_Chance = sConfigMgr->GetOption<int>("OllamaChat.EventTypeDefeated_Chance", 0);
    g_EventTypeDefeatedPlayer_Chance = sConfigMgr->GetOption<int>("OllamaChat.EventTypeDefeatedPlayer_Chance", 0);
    g_EventTypePetDefeated_Chance = sConfigMgr->GetOption<int>("OllamaChat.EventTypePetDefeated_Chance", 0);
    g_EventTypeGotItem_Chance = sConfigMgr->GetOption<int>("OllamaChat.EventTypeGotItem_Chance", 0);
    g_EventTypeDied_Chance = sConfigMgr->GetOption<int>("OllamaChat.EventTypeDied_Chance", 0);
    g_EventTypeCompletedQuest_Chance = sConfigMgr->GetOption<int>("OllamaChat.EventTypeCompletedQuest_Chance", 0);
    g_EventTypeLearnedSpell_Chance = sConfigMgr->GetOption<int>("OllamaChat.EventTypeLearnedSpell_Chance", 0);
    g_EventTypeRequestedDuel_Chance = sConfigMgr->GetOption<int>("OllamaChat.EventTypeRequestedDuel_Chance", 0);
    g_EventTypeStartedDueling_Chance = sConfigMgr->GetOption<int>("OllamaChat.EventTypeStartedDueling_Chance", 0);
    g_EventTypeWonDuel_Chance = sConfigMgr->GetOption<int>("OllamaChat.EventTypeWonDuel_Chance", 0);
    g_EventTypeLeveledUp_Chance = sConfigMgr->GetOption<int>("OllamaChat.EventTypeLeveledUp_Chance", 0);
    g_EventTypeAchievement_Chance = sConfigMgr->GetOption<int>("OllamaChat.EventTypeAchievement_Chance", 0);
    g_EventTypeUsedObject_Chance = sConfigMgr->GetOption<int>("OllamaChat.EventTypeUsedObject_Chance", 0);

    // Load chance variables for guild events
    g_GuildEventTypeEpicGear_Chance = sConfigMgr->GetOption<int>("OllamaChat.GuildEventTypeEpicGear_Chance", 0);
    g_GuildEventTypeRareGear_Chance = sConfigMgr->GetOption<int>("OllamaChat.GuildEventTypeRareGear_Chance", 0);
    g_GuildEventTypeGuildJoin_Chance = sConfigMgr->GetOption<int>("OllamaChat.GuildEventTypeGuildJoin_Chance", 0);
    g_GuildEventTypeGuildLogin_Chance = sConfigMgr->GetOption<int>("OllamaChat.GuildEventTypeGuildLogin_Chance", 0);
    g_GuildEventTypeGuildLeave_Chance = sConfigMgr->GetOption<int>("OllamaChat.GuildEventTypeGuildLeave_Chance", 0);
    g_GuildEventTypeGuildPromotion_Chance = sConfigMgr->GetOption<int>("OllamaChat.GuildEventTypeGuildPromotion_Chance", 0);
    g_GuildEventTypeGuildDemotion_Chance = sConfigMgr->GetOption<int>("OllamaChat.GuildEventTypeGuildDemotion_Chance", 0);
    g_GuildEventTypeGuildAchievement_Chance = sConfigMgr->GetOption<int>("OllamaChat.GuildEventTypeGuildAchievement_Chance", 0);
    g_GuildEventTypeLevelUp_Chance = sConfigMgr->GetOption<int>("OllamaChat.GuildEventTypeLevelUp_Chance", 0);
    g_GuildEventTypeDungeonComplete_Chance = sConfigMgr->GetOption<int>("OllamaChat.GuildEventTypeDungeonComplete_Chance", 0);


    // Cooldown time for events
    g_EventCooldownTime = sConfigMgr->GetOption<uint32_t>("OllamaChat.EventCooldownTime", 10);

    // Channel disable settings
    g_DisableForCustomChannels = sConfigMgr->GetOption<bool>("OllamaChat.DisableForCustomChannels", false);
    g_DisableForSayYell = sConfigMgr->GetOption<bool>("OllamaChat.DisableForSayYell", false);
    g_DisableForGuild = sConfigMgr->GetOption<bool>("OllamaChat.DisableForGuild", false);
    g_DisableForParty = sConfigMgr->GetOption<bool>("OllamaChat.DisableForParty", false);

    LOG_INFO("server.loading",
             "[Ollama Chat] Config loaded: Enabled = {}, SayDistance = {}, YellDistance = {}, "
             "Reply Chances - Say: P{}%/B{}%, Channel: P{}%/B{}%, Party: P{}%/B{}%, Guild: P{}%/B{}%, MaxBotsToPick = {}, "
             "Url = {}, Model = {}, MaxConcurrentQueries = {}, EnableRandomChatter = {}, MinRandInt = {}, MaxRandInt = {}, RandomChatterRealPlayerDistance = {}, "
             "RandomChatterBotCommentChance = {}. MaxConcurrentQueries = {}. Extra blacklist commands: {}",
             g_Enable, g_SayDistance, g_YellDistance,
             g_PlayerReplyChance_Say, g_BotReplyChance_Say,
             g_PlayerReplyChance_Channel, g_BotReplyChance_Channel,
             g_PlayerReplyChance_Party, g_BotReplyChance_Party,
             g_PlayerReplyChance_Guild, g_BotReplyChance_Guild,
             g_MaxBotsToPick,
             g_OllamaUrl, g_OllamaModel, g_MaxConcurrentQueries,
             g_EnableRandomChatter, g_MinRandomInterval, g_MaxRandomInterval, g_RandomChatterRealPlayerDistance,
             g_RandomChatterBotCommentChance, g_MaxConcurrentQueries, extraBlacklist);

    if (g_DebugEnabled)
    {
        SpeechSplitSelfTest();
        SoftStopSelfTest();
    }
}

void LoadPersonalityTemplatesFromDB()
{
    g_PersonalityPrompts.clear();
    g_PersonalityKeys.clear();
    g_PersonalityKeysRandomOnly.clear();

    QueryResult result = CharacterDatabase.Query("SELECT `key`, `prompt`, `manual_only` FROM `mod_ollama_chat_personality_templates`");
    if (!result)
    {
        LOG_ERROR("server.loading", "[Ollama Chat] No personality templates found in the database!");
        return;
    }

    do
    {
        std::string key = (*result)[0].Get<std::string>();
        std::string prompt = (*result)[1].Get<std::string>();
        bool manualOnly = (*result)[2].Get<bool>();
        
        g_PersonalityPrompts[key] = prompt;
        g_PersonalityKeys.push_back(key);
        
        // Only add to random pool if not manual_only
        if (!manualOnly)
        {
            g_PersonalityKeysRandomOnly.push_back(key);
        }
    } while (result->NextRow());

    LOG_INFO("server.loading", "[Ollama Chat] Cached {} personalities ({} available for random assignment).", 
             g_PersonalityKeys.size(), g_PersonalityKeysRandomOnly.size());
}

void LoadBotConversationHistoryFromDB()
{
    QueryResult result = CharacterDatabase.Query(
        "SELECT bot_guid, player_guid, player_message, bot_reply FROM mod_ollama_chat_history ORDER BY timestamp ASC"
    );
    if (!result)
        return;

    std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
    g_BotConversationHistory.clear();

    do {
        uint64_t botGuid = (*result)[0].Get<uint64_t>();
        uint64_t playerGuid = (*result)[1].Get<uint64_t>();
        std::string playerMsg = (*result)[2].Get<std::string>();
        std::string botReply = (*result)[3].Get<std::string>();

        auto& playerHistory = g_BotConversationHistory[botGuid][playerGuid];
        playerHistory.push_back({ playerMsg, botReply });
        while (playerHistory.size() > g_MaxConversationHistory)
        {
            playerHistory.pop_front();
        }

    } while (result->NextRow());

}


// Definition of the configuration WorldScript.
OllamaChatConfigWorldScript::OllamaChatConfigWorldScript() : WorldScript("OllamaChatConfigWorldScript") { }

void OllamaChatConfigWorldScript::OnStartup()
{
    LoadOllamaChatConfig();
    LoadBotPersonalityList();
    LoadBotConversationHistoryFromDB();
    InitializeSentimentTracking();

    // Initialize RAG system if enabled
    if (g_EnableRAG) {
        if (g_RAGSystem) {
            delete g_RAGSystem;
        }
        g_RAGSystem = new OllamaRAGSystem();
        if (!g_RAGSystem->Initialize()) {
            LOG_ERROR("server.loading", "[Ollama Chat] Failed to initialize RAG system");
            delete g_RAGSystem;
            g_RAGSystem = nullptr;
        } else {
            LOG_INFO("server.loading", "[Ollama Chat] RAG system initialized successfully");
        }
    }
}

void OllamaChatConfigWorldScript::OnShutdown()
{
    // Clean up RAG system
    if (g_RAGSystem) {
        delete g_RAGSystem;
        g_RAGSystem = nullptr;
        LOG_INFO("server.loading", "[Ollama Chat] RAG system cleaned up");
    }
}
