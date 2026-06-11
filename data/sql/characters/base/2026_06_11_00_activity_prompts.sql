CREATE TABLE IF NOT EXISTS `mod_ollama_chat_activity_prompts` (
  `activity`  VARCHAR(32)  NOT NULL,
  `lifecycle` VARCHAR(16)  NOT NULL,
  `prompt`    TEXT         NOT NULL,
  PRIMARY KEY (`activity`, `lifecycle`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO `mod_ollama_chat_activity_prompts` (`activity`,`lifecycle`,`prompt`) VALUES
('fishing','start',  'You''ve just settled in to fish here. React in under 15 words, in character — the quiet, the wait, the water.'),
('fishing','finish', 'You''ve just finished fishing here. React in under 15 words, in character — your catch, or what''s next.')
ON DUPLICATE KEY UPDATE `prompt` = VALUES(`prompt`);
