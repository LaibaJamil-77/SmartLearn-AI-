CREATE DATABASE smartlearn_ai;
USE smartlearn_ai;

-- Users table for authentication and profile management
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL, -- Hashed password
    avatar_index INT, -- 0 or 1, matching avatars array in Head.jsx
    first_name VARCHAR(50), -- From signup form
    last_name VARCHAR(50), -- From signup form
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Topics table for topic-based slide generation
CREATE TABLE topics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    topic_name VARCHAR(255) NOT NULL, -- e.g., "Photosynthesis"
    instructions TEXT, -- Optional instructions
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Documents table for uploaded files
CREATE TABLE documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    filename VARCHAR(255) NOT NULL, -- e.g., "notes.pdf"
    instructions TEXT, -- Optional instructions
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Slides table for generated presentations
CREATE TABLE slides (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    topic_id INT NULL, -- Links to topics if topic-based
    document_id INT NULL, -- Links to documents if document-based
    ppt_filename VARCHAR(255) NOT NULL, -- e.g., "ppt_files/topic_testuser_20250417_0919.pptx"
    slide_count INT NOT NULL, -- Number of slides
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE SET NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL
);

-- Slide Content table for individual slide details
CREATE TABLE slide_content (
    id INT AUTO_INCREMENT PRIMARY KEY,
    slide_id INT NOT NULL,
    slide_number INT NOT NULL, -- 1, 2, 3, etc.
    title VARCHAR(255) NOT NULL, -- Slide title
    content TEXT NOT NULL, -- Slide body
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (slide_id) REFERENCES slides(id) ON DELETE CASCADE,
    UNIQUE (slide_id, slide_number) -- Ensures no duplicate slide numbers
);

-- QA History table for question-answer interactions
CREATE TABLE qa_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    slide_id INT NULL, -- Links to slides for context
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    context_type ENUM('topic', 'document', 'general') NOT NULL,
    response_language ENUM('English', 'Roman Urdu') NOT NULL,
    asked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (slide_id) REFERENCES slides(id) ON DELETE SET NULL
);

-- User Progress table for tracking learning progress
CREATE TABLE user_progress (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    slide_id INT NULL, -- Links to completed slide sets
    topic_id INT NULL, -- Links to topics
    document_id INT NULL, -- Links to documents
    progress_status ENUM('started', 'in_progress', 'completed') NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (slide_id) REFERENCES slides(id) ON DELETE SET NULL,
    FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE SET NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL
);
ALTER TABLE slide_content ADD COLUMN explanation TEXT;