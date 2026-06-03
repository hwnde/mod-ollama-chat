CREATE TABLE IF NOT EXISTS `mod_ollama_chat_npc` (
  `entry` INT UNSIGNED NOT NULL PRIMARY KEY,
  `cooldown_sec` INT UNSIGNED NULL,
  `persona_hint` TEXT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO `mod_ollama_chat_npc` (`entry`, `cooldown_sec`, `persona_hint`) VALUES
  (4949,  NULL, NULL),   -- Thrall (Orgrimmar)
  (17852, NULL, NULL),   -- Thrall (later)
  (18063, NULL, NULL),   -- Garrosh Hellscream
  (25237, NULL, NULL),   -- Garrosh Hellscream (Warchief)
  (10181, NULL, NULL),   -- Lady Sylvanas Windrunner
  (37596, NULL, NULL),   -- Lady Sylvanas Windrunner (later)
  (3057,  NULL, NULL),   -- Cairne Bloodhoof
  (16802, NULL, NULL),   -- Lor'themar Theron
  (21984, NULL, NULL),   -- Rexxar
  (29611, NULL, NULL),   -- King Varian Wrynn
  (32401, NULL, NULL),   -- King Varian Wrynn (later)
  (4968,  NULL, NULL),   -- Lady Jaina Proudmoore
  (32346, NULL, NULL),   -- Lady Jaina Proudmoore (later)
  (7937,  NULL, NULL),   -- High Tinker Mekkatorque
  (17948, NULL, NULL),   -- Tyrande Whisperwind
  (3936,  NULL, NULL),   -- Shandris Feathermoon
  (16128, NULL, NULL),   -- Rhonin
  (27990, NULL, NULL),   -- Krasus
  (26917, NULL, NULL),   -- Alexstrasza the Life-Binder
  (31333, NULL, NULL),   -- Alexstrasza the Life-Binder (later)
  (10162, NULL, NULL)    -- Lord Victor Nefarius
ON DUPLICATE KEY UPDATE `entry` = `entry`;
