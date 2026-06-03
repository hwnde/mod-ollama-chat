#include "mod-ollama-chat_worldnpc.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat-utilities.h"   // SafeFormat
#include "mod-ollama-chat_api.h"         // QueryOllamaAPI
#include "mod-ollama-chat_rag.h"         // g_RAGSystem, OllamaRAGSystem
#include "QuestDef.h"                     // QuestGiverStatus / DIALOG_STATUS_*
#include "UnitDefines.h"                  // UNIT_NPC_FLAG_*
#include "SharedDefines.h"                // RACE_*, CLASS_*
#include "Log.h"
#include "Player.h"
#include "Creature.h"
#include "ObjectAccessor.h"
#include "PlayerbotAI.h"
#include "PlayerbotMgr.h"
#include "Cell.h"
#include "CellImpl.h"
#include "GridNotifiers.h"
#include "GridNotifiersImpl.h"
#include "DBCStores.h"
#include "World.h"
#include "Timer.h"
#include "Random.h"
#include <vector>
#include <string>
#include <list>
#include <unordered_map>
#include <thread>
#include <chrono>

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

// Pure: assemble the persona prompt for a curated NPC character. ragInfo/personaHint may be empty.
static std::string BuildNpcCharacterPrompt(const std::string& name, const std::string& title,
                                           const std::string& zone, const std::string& race,
                                           const std::string& cls, const std::string& personaHint,
                                           const std::string& ragInfo)
{
    std::string p = SafeFormat(g_WorldNpcCharacterPrompt,
                               fmt::arg("name", name),
                               fmt::arg("title", title.empty() ? "a figure of note" : title),
                               fmt::arg("zone", zone),
                               fmt::arg("race", race),
                               fmt::arg("class", cls));
    if (!personaHint.empty())
        p += " " + personaHint;
    if (!ragInfo.empty())
        p += "\n" + SafeFormat(g_RAGPromptTemplate, fmt::arg("rag_info", ragInfo));
    return p;
}

// Global NPC LLM-call budget: at most g_WorldNpcCharacterCallsPerMin calls per rolling minute.
// Only called from the world thread (HandleNpcProximityChatter) — static counters are safe.
static bool ConsumeNpcLlmBudget()
{
    static uint32 windowStartMs = 0;
    static uint32 windowCount   = 0;
    uint32 now = getMSTime();
    if (windowStartMs == 0 || GetMSTimeDiffToNow(windowStartMs) >= 60000)
    {
        windowStartMs = now;
        windowCount   = 0;
    }
    if (windowCount >= g_WorldNpcCharacterCallsPerMin)
        return false;
    ++windowCount;
    return true;
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

namespace
{
    // Matches any creature within range.  operator() takes Unit* to satisfy
    // Acore::CreatureListSearcher (which drives AllCreaturesOfEntryInRange pattern).
    class AnyCreatureInRangeCheck
    {
    public:
        AnyCreatureInRangeCheck(WorldObject const* obj, float range) : _obj(obj), _range(range) {}
        bool operator()(Unit* u) { return u && _obj->IsWithinDist(u, _range, false); }
    private:
        WorldObject const* _obj;
        float _range;
    };

    // guid raw value -> last-bark msTime
    std::unordered_map<uint64_t, uint32> s_npcCooldown;
    std::unordered_map<uint64_t, uint32> s_playerCooldown;
}

void OllamaWorldNpcChatter::HandleNpcProximityChatter()
{
    uint32 nowMs = getMSTime();
    uint32 npcCdMs    = g_WorldNpcChatNpcCooldownSec    * 1000;
    uint32 playerCdMs = g_WorldNpcChatPlayerCooldownSec * 1000;

    for (auto const& pair : ObjectAccessor::GetPlayers())
    {
        Player* player = pair.second;
        if (!player || !player->IsInWorld())
            continue;
        // real players only -- bots return non-null here
        if (PlayerbotsMgr::instance().GetPlayerbotAI(player))
            continue;

        uint64_t pguid = player->GetGUID().GetRawValue();
        auto pit = s_playerCooldown.find(pguid);
        if (pit != s_playerCooldown.end() && GetMSTimeDiffToNow(pit->second) < playerCdMs)
            continue;

        std::list<Creature*> nearby;
        AnyCreatureInRangeCheck check(player, g_WorldNpcChatRange);
        Acore::CreatureListSearcher<AnyCreatureInRangeCheck> searcher(player, nearby, check);
        Cell::VisitObjects(player, searcher, g_WorldNpcChatRange);
        if (nearby.empty())
            continue;

        nearby.sort([player](Creature* a, Creature* b) {
            return player->GetDistance(a) < player->GetDistance(b);
        });

        std::string zone;
        if (AreaTableEntry const* z = sAreaTableStore.LookupEntry(player->GetZoneId()))
        {
            zone = z->area_name[sWorld->GetDefaultDbcLocale()];
            if (zone.empty())
                zone = z->area_name[LOCALE_enUS];
        }

        uint32 emitted = 0;
        for (Creature* c : nearby)
        {
            if (emitted >= g_WorldNpcChatMaxPerTick)
                break;
            if (!c || !c->IsAlive() || c->IsInCombat())
                continue;
            if (!c->IsFriendlyTo(player))
                continue;
            NPCFlags flags = c->GetNpcFlags();
            if (!flags)
                continue;

            uint64_t cguid = c->GetGUID().GetRawValue();
            auto nit = s_npcCooldown.find(cguid);
            if (nit != s_npcCooldown.end() && GetMSTimeDiffToNow(nit->second) < npcCdMs)
                continue;

            // ---- Tier 1: curated LLM character (precedence over role barks) ----
            if (g_WorldNpcCharactersEnable && !g_WorldNpcCharacters.empty())
            {
                auto cit = g_WorldNpcCharacters.find(c->GetEntry());
                if (cit != g_WorldNpcCharacters.end())
                {
                    uint32 cdSec = cit->second.cooldownSec ? cit->second.cooldownSec : g_WorldNpcCharacterCooldownSec;
                    auto nit2 = s_npcCooldown.find(cguid);
                    if (nit2 != s_npcCooldown.end() && GetMSTimeDiffToNow(nit2->second) < cdSec * 1000u)
                    {
                        continue;   // on cooldown -> stay silent (do NOT fall to role bark)
                    }
                    if (!ConsumeNpcLlmBudget())
                    {
                        continue;   // global budget spent -> skip (no role-bark fallback)
                    }

                    std::string title;
                    CreatureTemplate const* ct = c->GetCreatureTemplate();
                    if (ct)
                        title = ct->SubName;

                    std::string ragInfo;
                    if (g_EnableRAGInitiated && g_RAGSystem)
                    {
                        std::string q = c->GetName() + " " + zone + (title.empty() ? "" : (" " + title));
                        auto rr = g_RAGSystem->RetrieveRelevantInfo(q, g_RAGInitiatedMaxItems, g_RAGSimilarityThreshold);
                        ragInfo = g_RAGSystem->GetFormattedRAGInfo(rr);
                    }

                    std::string prompt = BuildNpcCharacterPrompt(
                        c->GetName(), title, zone,
                        RaceName(player->getRace()), ClassName(player->getClass()),
                        cit->second.personaHint, ragInfo);

                    ObjectGuid pg = player->GetGUID();
                    ObjectGuid cg = c->GetGUID();
                    float range = g_WorldNpcChatRange;

                    std::thread([pg, cg, prompt, range]() {
                        try {
                            std::string response = QueryOllamaAPI(prompt);
                            if (response.empty())
                                return;
                            Player* p = ObjectAccessor::FindPlayer(pg);
                            if (!p || !p->IsInWorld())
                                return;
                            Creature* cr = ObjectAccessor::GetCreature(*p, cg);
                            if (!cr || !cr->IsInWorld() || !cr->IsAlive())
                                return;
                            if (p->GetDistance(cr) > range + 10.0f)
                                return;   // player wandered off
                            std::vector<std::string> lines = SplitChatResponse(response);
                            for (size_t i = 0; i < lines.size(); ++i)
                            {
                                if (i > 0 && g_SpeechSplitLineDelayMs > 0)
                                    std::this_thread::sleep_for(std::chrono::milliseconds(g_SpeechSplitLineDelayMs));
                                cr->Say(lines[i], LANG_UNIVERSAL, p);
                            }
                        } catch (const std::exception& e) {
                            LOG_ERROR("server.loading", "[Ollama Chat] WorldNpc character thread: {}", e.what());
                        } catch (...) {}
                    }).detach();

                    s_npcCooldown[cguid]    = nowMs;
                    s_playerCooldown[pguid] = nowMs;
                    ++emitted;
                    if (g_DebugEnabled)
                        LOG_INFO("server.loading", "[Ollama Chat] WorldNpc character {} addressed {} (LLM dispatched).",
                                 c->GetName(), player->GetName());
                    continue;   // curated -> never also role-bark
                }
            }
            // ---- Tier 2: role barks (existing P1 code follows) ----

            QuestGiverStatus questStatus = DIALOG_STATUS_NONE;
            if (flags & UNIT_NPC_FLAG_QUESTGIVER)
                questStatus = player->GetQuestDialogStatus(c);

            WorldNpcRole role = ResolveNpcRole(static_cast<uint32>(flags), static_cast<uint32>(questStatus));
            if (role == WN_NONE)
                continue;
            std::vector<std::string> const& pool = PhrasesForRole(role);
            if (pool.empty())
                continue;

            std::string templ = pool.size() == 1 ? pool[0] : pool[urand(0, static_cast<uint32>(pool.size() - 1))];
            std::string line = FillPhrase(templ, c->GetName(),
                                          RaceName(player->getRace()), ClassName(player->getClass()),
                                          zone, player->GetName());
            if (line.empty() || line == "[Format Error]")
                continue;

            c->Say(line, LANG_UNIVERSAL, player);
            s_npcCooldown[cguid]  = nowMs;
            s_playerCooldown[pguid] = nowMs;
            ++emitted;

            if (g_DebugEnabled)
                LOG_INFO("server.loading", "[Ollama Chat] WorldNpc {} (role {}) greeted {}: {}",
                         c->GetName(), static_cast<int>(role), player->GetName(), line);
        }
    }

    // bound the cooldown maps: drop entries already past their cooldown
    for (auto it = s_npcCooldown.begin(); it != s_npcCooldown.end(); )
        it = (GetMSTimeDiffToNow(it->second) >= npcCdMs) ? s_npcCooldown.erase(it) : std::next(it);
    for (auto it = s_playerCooldown.begin(); it != s_playerCooldown.end(); )
        it = (GetMSTimeDiffToNow(it->second) >= playerCdMs) ? s_playerCooldown.erase(it) : std::next(it);
}
