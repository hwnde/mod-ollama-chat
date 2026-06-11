-- occupation-lifecycle-chat: per-row chance + re-key fishing->fish + seed all 32 lifecycle fragments.
-- Runs once (dbupdater hash-tracked) -> plain ALTER is safe on both MySQL and MariaDB.
-- The Phase-1 base file seeds the 2 'fishing' rows in a 3-col table; this update evolves it.

ALTER TABLE `mod_ollama_chat_activity_prompts`
    ADD COLUMN `chance` INT UNSIGNED NOT NULL DEFAULT 100;

-- migrate the two Phase-1 rows to the BehaviorKey() key (BEH_FISH -> "fish"):
UPDATE `mod_ollama_chat_activity_prompts` SET `activity` = 'fish' WHERE `activity` = 'fishing';

-- Seed all 32 lifecycle fragments (16 behaviors x {start,finish}); idempotent upsert.
-- `chance` is the per-row talk chance (0-100 percent); tuned by frequency: frequent behaviors low,
-- rare/interesting high. The 2 fish rows are upserted here too so they carry an intentional chance.
INSERT INTO `mod_ollama_chat_activity_prompts` (`activity`,`lifecycle`,`prompt`,`chance`) VALUES
 ('fish','start','You have just settled in to fish here. React in under 15 words, in character - the quiet, the wait, the water.',20),
 ('fish','finish','You have just finished fishing here. React in under 15 words, in character - your catch, or what comes next.',18),
 ('go_grind','start','You are setting out to thin some hostile creatures. Say one short, in-character line as you head off.',20),
 ('go_grind','finish','You have killed your fill for now. Say one short, in-character line wrapping it up.',15),
 ('wander_random','start','You are idly wandering to see what is nearby. Say one short, in-character line of idle curiosity.',8),
 ('wander_random','finish','You are drifting back from your wander. Say one short, in-character line.',6),
 ('do_quest','start','You are taking on a new task. Say one short, in-character line about setting to it.',25),
 ('do_quest','finish','You have finished a task and earned your due. Say one short, in-character line.',25),
 ('gathering_circuit','start','You are reading the land for nodes to harvest. Say one short, in-character line.',20),
 ('gathering_circuit','finish','Your bags are filling from the gathering. Say one short, in-character line.',18),
 ('rest','start','You are settling in at an inn for a breather. Say one short, in-character line.',12),
 ('rest','finish','You are getting up, rested, and back to it. Say one short, in-character line.',10),
 ('wander_npc','start','You are approaching some folk to greet them. Say one short, in-character line.',15),
 ('wander_npc','finish','You are parting from the folk you greeted. Say one short, in-character line.',12),
 ('travel_flight','start','You are boarding a gryphon or wyvern for a long flight. Say one short, in-character line.',30),
 ('travel_flight','finish','You have just touched down from your flight. Say one short, in-character line.',25),
 ('travel_mount','start','You have a long road ahead on your mount. Say one short, in-character line.',12),
 ('travel_mount','finish','You have arrived at the hub after a long ride. Say one short, in-character line.',12),
 ('outdoor_pvp','start','You have spotted a contested point worth fighting over. Say one short, in-character line.',50),
 ('outdoor_pvp','finish','The fight over the point is settled. Say one short, in-character line about how it went.',45),
 ('social','start','You are striking up company with someone. Say one short, in-character line.',15),
 ('social','finish','You are bidding farewell to the company. Say one short, in-character line.',12),
 ('loiter','start','You are settling into the tavern, forge, or bank for a spell. Say one short, in-character line.',8),
 ('loiter','finish','You are moving on from where you lingered. Say one short, in-character line.',6),
 ('craft','start','You are sitting down to your craft. Say one short, in-character line.',25),
 ('craft','finish','You have finished a fine piece of work. Say one short, in-character line.',25),
 ('duel','start','You have issued or accepted a duel and salute your foe. Say one short, in-character line.',60),
 ('duel','finish','The duel is over. Say one short, in-character line of banter about the result.',55),
 ('repair_sell','start','You need to offload some junk and patch your gear. Say one short, in-character line.',12),
 ('repair_sell','finish','Your gear is good as new and your bags lighter. Say one short, in-character line.',10),
 ('dummy','start','You are warming up against a training dummy. Say one short, in-character line.',15),
 ('dummy','finish','You are done drilling on the dummy. Say one short, in-character line.',12)
ON DUPLICATE KEY UPDATE `prompt` = VALUES(`prompt`), `chance` = VALUES(`chance`);
