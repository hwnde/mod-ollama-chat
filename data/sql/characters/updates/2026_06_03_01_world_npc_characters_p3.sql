-- world-npc-llm-chat P3: promote the rag-notable-figures roster into mod_ollama_chat_npc (Tier-1 characters).
-- Applies after base/2026_06_03_00_world_npc_chat.sql (table + 5 iconic + other seed).
-- Entry IDs resolved against creature_template: friendly, titled, world-present, npcflag>0.
-- Idempotent: ON DUPLICATE KEY UPDATE entry=entry is a no-op (preserves any operator edits to cooldown_sec/persona_hint).
INSERT INTO `mod_ollama_chat_npc` (`entry`, `cooldown_sec`, `persona_hint`) VALUES
  -- Horde (10)
  (14720, NULL, NULL),   -- High Overlord Saurfang (Varok)
  (11878, NULL, NULL),   -- Nathanos Blightcaller
  (3230,  NULL, NULL),   -- Nazgrel, Advisor to Thrall
  (4046,  NULL, NULL),   -- Magatha Grimtotem
  (3144,  NULL, NULL),   -- Eitrigg
  (2993,  NULL, NULL),   -- Baine Bloodhoof
  (37527, NULL, NULL),   -- Halduron Brightwing (flagged variant)
  (17076, NULL, NULL),   -- Lady Liadrin (flagged variant)
  (30116, NULL, NULL),   -- Archmage Aethas Sunreaver
  (26581, NULL, NULL),   -- Koltira Deathweaver
  -- Alliance (5)
  (1747,  NULL, NULL),   -- Anduin Wrynn
  (5635,  NULL, NULL),   -- Falstad Wildhammer
  (30115, NULL, NULL),   -- Vereesa Windrunner
  (8929,  NULL, NULL),   -- Princess Moira Bronzebeard
  (26170, NULL, NULL),   -- Thassarian
  -- Neutral (3)
  (5769,  NULL, NULL),   -- Arch Druid Hamuul Runetotem
  (11034, NULL, NULL),   -- Lord Maxwell Tyrosus
  (17204, NULL, NULL)    -- Farseer Nobundo
ON DUPLICATE KEY UPDATE `entry` = `entry`;

-- Dropped from the roster (no usable creature_template row): Drek'Thar (all variants npcflag=0),
-- Grand Magister Rommath (npcflag=0), Dranosh Saurfang (no friendly row). Their RAG entries
-- still ground bot chat via rag-notable-figures; they cannot Tier-1-speak (scan skips npcflag==0).
-- The 5 iconic figures (Shandris/Rexxar/Krasus/Lord Victor Nefarius/Alexstrasza) are already in
-- base/2026_06_03_00_world_npc_chat.sql and are NOT re-added here.
