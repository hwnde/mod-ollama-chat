#include "mod-ollama-chat_worldnpc.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat-utilities.h"   // SafeFormat
#include "QuestDef.h"                     // QuestGiverStatus / DIALOG_STATUS_*
#include "UnitDefines.h"                  // UNIT_NPC_FLAG_*
#include "SharedDefines.h"                // RACE_*, CLASS_*
#include "Log.h"
#include <vector>
#include <string>

// ---- role model -----------------------------------------------------------
enum WorldNpcRole
{
    WN_NONE = 0,
    WN_QUESTGIVER_AVAILABLE,
    WN_QUESTGIVER_TURNIN,
    WN_INNKEEPER,
    WN_VENDOR,
    WN_FLIGHTMASTER,
    WN_TRAINER,
    WN_BANKER
};

// Pure: resolve an npcflag bitmask (+ quest status for questgivers) to a role.
// Priority: an actionable questgiver first, then service roles.
static WorldNpcRole ResolveNpcRole(uint32 flags, uint32 questStatus)
{
    if (flags & UNIT_NPC_FLAG_QUESTGIVER)
    {
        if (questStatus == DIALOG_STATUS_AVAILABLE         ||
            questStatus == DIALOG_STATUS_AVAILABLE_REP     ||
            questStatus == DIALOG_STATUS_LOW_LEVEL_AVAILABLE    ||
            questStatus == DIALOG_STATUS_LOW_LEVEL_AVAILABLE_REP)
            return WN_QUESTGIVER_AVAILABLE;
        if (questStatus == DIALOG_STATUS_REWARD            ||
            questStatus == DIALOG_STATUS_REWARD2           ||
            questStatus == DIALOG_STATUS_REWARD_REP        ||
            questStatus == DIALOG_STATUS_LOW_LEVEL_REWARD_REP)
            return WN_QUESTGIVER_TURNIN;
        // questgiver with nothing for this player -> fall through to other roles
    }
    if (flags & UNIT_NPC_FLAG_INNKEEPER)    return WN_INNKEEPER;
    if (flags & UNIT_NPC_FLAG_VENDOR)       return WN_VENDOR;
    if (flags & UNIT_NPC_FLAG_FLIGHTMASTER) return WN_FLIGHTMASTER;
    if (flags & UNIT_NPC_FLAG_TRAINER)      return WN_TRAINER;
    if (flags & UNIT_NPC_FLAG_BANKER)       return WN_BANKER;
    return WN_NONE;
}

static std::vector<std::string> const& PhrasesForRole(WorldNpcRole r)
{
    static const std::vector<std::string> empty;
    switch (r)
    {
        case WN_QUESTGIVER_AVAILABLE: return g_WorldNpcPhrasesQuestgiver;
        case WN_QUESTGIVER_TURNIN:    return g_WorldNpcPhrasesQuestTurnin;
        case WN_INNKEEPER:            return g_WorldNpcPhrasesInnkeeper;
        case WN_VENDOR:               return g_WorldNpcPhrasesVendor;
        case WN_FLIGHTMASTER:         return g_WorldNpcPhrasesFlightmaster;
        case WN_TRAINER:              return g_WorldNpcPhrasesTrainer;
        case WN_BANKER:               return g_WorldNpcPhrasesBanker;
        default:                      return empty;
    }
}

static std::string RaceName(uint8 race)
{
    switch (race)
    {
        case RACE_HUMAN:            return "human";
        case RACE_ORC:              return "orc";
        case RACE_DWARF:            return "dwarf";
        case RACE_NIGHTELF:         return "night elf";
        case RACE_UNDEAD_PLAYER:    return "forsaken";
        case RACE_TAUREN:           return "tauren";
        case RACE_GNOME:            return "gnome";
        case RACE_TROLL:            return "troll";
        case RACE_BLOODELF:         return "blood elf";
        case RACE_DRAENEI:          return "draenei";
        default:                    return "traveler";
    }
}

static std::string ClassName(uint8 cls)
{
    switch (cls)
    {
        case CLASS_WARRIOR:      return "warrior";
        case CLASS_PALADIN:      return "paladin";
        case CLASS_HUNTER:       return "hunter";
        case CLASS_ROGUE:        return "rogue";
        case CLASS_PRIEST:       return "priest";
        case CLASS_DEATH_KNIGHT: return "death knight";
        case CLASS_SHAMAN:       return "shaman";
        case CLASS_MAGE:         return "mage";
        case CLASS_WARLOCK:      return "warlock";
        case CLASS_DRUID:        return "druid";
        default:                 return "adventurer";
    }
}

// Pure: fill a phrase template's placeholders.
static std::string FillPhrase(const std::string& templ, const std::string& npcName,
                              const std::string& race, const std::string& cls,
                              const std::string& zone, const std::string& playerName)
{
    return SafeFormat(templ,
                      fmt::arg("name", npcName),
                      fmt::arg("race", race),
                      fmt::arg("class", cls),
                      fmt::arg("zone", zone),
                      fmt::arg("player", playerName));
}

void WorldNpcChatSelfTest()
{
    int passed = 0, total = 0;
    auto check = [&](const char* nm, bool ok) {
        ++total;
        if (ok) ++passed;
        else LOG_ERROR("server.loading", "[Ollama Chat] WorldNpcChat self-test FAIL ({})", nm);
    };

    check("qg-available", ResolveNpcRole(UNIT_NPC_FLAG_QUESTGIVER, DIALOG_STATUS_AVAILABLE) == WN_QUESTGIVER_AVAILABLE);
    check("qg-turnin",    ResolveNpcRole(UNIT_NPC_FLAG_QUESTGIVER, DIALOG_STATUS_REWARD)    == WN_QUESTGIVER_TURNIN);
    check("qg-none->vendor", ResolveNpcRole(UNIT_NPC_FLAG_QUESTGIVER | UNIT_NPC_FLAG_VENDOR, DIALOG_STATUS_NONE) == WN_VENDOR);
    check("qg-priority",  ResolveNpcRole(UNIT_NPC_FLAG_QUESTGIVER | UNIT_NPC_FLAG_INNKEEPER, DIALOG_STATUS_AVAILABLE) == WN_QUESTGIVER_AVAILABLE);
    check("innkeeper",    ResolveNpcRole(UNIT_NPC_FLAG_INNKEEPER, 0) == WN_INNKEEPER);
    check("vendor",       ResolveNpcRole(UNIT_NPC_FLAG_VENDOR, 0) == WN_VENDOR);
    check("banker",       ResolveNpcRole(UNIT_NPC_FLAG_BANKER, 0) == WN_BANKER);
    check("none",         ResolveNpcRole(0, 0) == WN_NONE);

    std::string s = FillPhrase("Rest here, {race} {class}, welcome to {zone}.",
                               "Innkeeper Bob", "orc", "warrior", "Durotar", "Hero");
    check("phrase-fill", s == "Rest here, orc warrior, welcome to Durotar.");

    LOG_INFO("server.loading", "[Ollama Chat] WorldNpcChat self-test: {}/{} passed", passed, total);
}

OllamaWorldNpcChatter::OllamaWorldNpcChatter() : WorldScript("OllamaWorldNpcChatter") {}

void OllamaWorldNpcChatter::OnUpdate(uint32 diff)
{
    if (!g_Enable || !g_WorldNpcChatEnable)
        return;

    static uint32_t timer = 0;
    if (timer <= diff)
    {
        timer = g_WorldNpcChatTickMs;
        HandleNpcProximityChatter();
    }
    else
    {
        timer -= diff;
    }
}

void OllamaWorldNpcChatter::HandleNpcProximityChatter()
{
    // Implemented in Task 4.
}
