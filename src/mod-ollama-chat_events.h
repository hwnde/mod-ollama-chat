#ifndef MOD_OLLAMA_CHAT_EVENTS_H
#define MOD_OLLAMA_CHAT_EVENTS_H

#include "ScriptMgr.h"
#include "Player.h"
#include "mod-ollama-chat_handler.h"
#include <string>

class OllamaBotEventChatter
{
public:
    // guildOverride lets guild hooks supply the guild explicitly, so dispatch does not depend on
    // source->GetGuild() (which is unreliable while a member is being added/removed).
    void DispatchGameEvent(Player* source, std::string type, std::string detail, Guild* guildOverride = nullptr);
    void QueueEvent(Player* bot, std::string type, std::string detail, std::string actorName, bool isGuildEvent = false);
    std::string BuildPrompt(Player* bot, std::string promptTemplate, std::string eventType, std::string eventDetail, std::string actorName, ChannelCategory channelCat);
};

class ChatOnKill : public PlayerScript
{
public:
    ChatOnKill();
    void OnPlayerCreatureKill(Player* killer, Creature* victim);
    void OnPlayerPVPKill(Player* killer, Player* killed);
    void OnPlayerCreatureKilledByPet(Player* owner, Creature* victim);
};

class ChatOnLoot : public PlayerScript
{
public:
    ChatOnLoot();
    void OnPlayerStoreNewItem(Player* player, Item* item, uint32 count);
};

class ChatOnDeath : public PlayerScript
{
public:
    ChatOnDeath();
    void OnPlayerJustDied(Player* player);
};

class ChatOnQuest : public PlayerScript
{
public:
    ChatOnQuest();
    void OnPlayerCompleteQuest(Player* player, Quest const* quest);
};

class ChatOnLearn : public PlayerScript
{
public:
    ChatOnLearn();
    void OnPlayerLearnSpell(Player* player, uint32 spellID);
};

class ChatOnDuel : public PlayerScript
{
public:
    ChatOnDuel();
    void OnPlayerDuelRequest(Player* target, Player* challenger);
    void OnPlayerDuelStart(Player* player1, Player* player2);
    void OnPlayerDuelEnd(Player* winner, Player* loser, DuelCompleteType type);
};

// Extra events:
class ChatOnLevelUp : public PlayerScript
{
public:
    ChatOnLevelUp();
    void OnPlayerLevelChanged(Player* player, uint8 oldLevel);
};

class ChatOnAchievement : public PlayerScript
{
public:
    ChatOnAchievement();
    void OnPlayerCompleteAchievement(Player* player, AchievementEntry const* achievement);
};

class ChatOnGameObjectUse : public PlayerScript
{
public:
    ChatOnGameObjectUse();
    void OnGameObjectUse(Player* player, GameObject* go);
};

// additional event-chatter hooks
class ChatOnZone : public PlayerScript
{
public:
    ChatOnZone();
    void OnPlayerUpdateZone(Player* player, uint32 newZone, uint32 newArea) override;
};

class ChatOnKilledByCreature : public PlayerScript
{
public:
    ChatOnKilledByCreature();
    void OnPlayerKilledByCreature(Creature* killer, Player* killed) override;
};

class ChatOnReputationRank : public PlayerScript
{
public:
    ChatOnReputationRank();
    void OnPlayerReputationRankChange(Player* player, uint32 factionID, ReputationRank /*newRank*/,
                                      ReputationRank /*oldRank*/, bool increased) override;
};

class ChatOnResurrect : public PlayerScript
{
public:
    ChatOnResurrect();
    void OnPlayerResurrect(Player* player, float restorePercent, bool& applySickness) override;
};

class ChatOnCombat : public PlayerScript
{
public:
    ChatOnCombat();
    void OnPlayerEnterCombat(Player* player, Unit* enemy) override;
    void OnPlayerLeaveCombat(Player* player) override;
};

// Subscribes to core bot activity lifecycle hooks (start/finish) and speaks a DB-driven line.
class OllamaActivityEventScript : public PlayerbotScript
{
public:
    OllamaActivityEventScript();
    void OnPlayerbotActivityStart(Player* bot, uint32 activity) override;
    void OnPlayerbotActivityFinish(Player* bot, uint32 activity) override;
};

// Guild membership events. AzerothCore exposes these on GuildScript (NOT PlayerScript):
// OnAddMember / OnRemoveMember cover join/leave (and hand us the Guild* + Player* directly),
// while OnEvent carries promote/demote via the guild event-log type.
class ChatOnGuildEvent : public GuildScript
{
public:
    ChatOnGuildEvent();
    void OnAddMember(Guild* guild, Player* player, uint8& plRank) override;
    void OnRemoveMember(Guild* guild, Player* player, bool isDisbanding, bool isKicked) override;
    void OnEvent(Guild* guild, uint8 eventType, ObjectGuid::LowType playerGuid1, ObjectGuid::LowType playerGuid2,
                 uint8 newRank) override;
};

// There is no guild-login hook; catch logins on PlayerScript and filter to guilded real players.
class ChatOnLogin : public PlayerScript
{
public:
    ChatOnLogin();
    void OnPlayerLogin(Player* player) override;
};


#endif // MOD_OLLAMA_CHAT_EVENTS_H
