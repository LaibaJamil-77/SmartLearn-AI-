import { useState, useEffect } from "react";
import { FaPlay, FaPause, FaChevronLeft, FaChevronRight, FaDownload, FaMicrophone, FaBars, FaComments, FaTimes, FaPaperclip, FaPaperPlane } from "react-icons/fa";
import { useNavigate } from 'react-router-dom';

import "./ProcessingView.css";
import HistoryPage from './HistoryPage';

const ProcessingView = ({
  processingComplete,
  slides,
  pptFilename,
  explanations,
  userInput,
  setUserInput,
  selectedLanguage,
  setSelectedLanguage,
  selectedAvatar,
  setCurrentView,
  onNewChat,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isAnswering, setIsAnswering] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const [currentSlideIndex, setCurrentSlideIndex] = useState(0);
  const [currentSlide, setCurrentSlide] = useState(slides && slides.length > 0 ? slides[0] : null);
  const [language, setLanguage] = useState(selectedLanguage);
  const [parsedSlides, setParsedSlides] = useState([]);
  const [sidebarExpanded, setSidebarExpanded] = useState(true);
  const [showChat, setShowChat] = useState(false);
  const [isChatPlaying, setIsChatPlaying] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showQuiz, setShowQuiz] = useState(false);
  const [quizQuestions, setQuizQuestions] = useState([]);
  const [quizLoading, setQuizLoading] = useState(false);
  const [quizAnswers, setQuizAnswers] = useState({});
  const [quizSubmitted, setQuizSubmitted] = useState(false);
  const [quizScore, setQuizScore] = useState(0);

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const isSpeechRecognitionSupported = !!SpeechRecognition;

  // Language codes for dropdown
  const languageOptions = [
    { code: 'en-US', label: 'English' },
    { code: 'es-ES', label: 'Spanish' },
    { code: 'de-DE', label: 'German' },
    { code: 'fr-FR', label: 'French' },
  ];

  // Language code mapping for LibreTranslate
  const langMap = {
    'en-US': 'en',
    'es-ES': 'es',
    'de-DE': 'de',
    'fr-FR': 'fr',
  };

  // Translation function using LibreTranslate
  async function translateText(text, targetLang) {
    if (targetLang === 'en') return text;
    try {
      const res = await fetch('https://libretranslate.de/translate', {
        method: 'POST',
        body: JSON.stringify({
          q: text,
          source: 'en',
          target: targetLang,
          format: 'text',
        }),
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await res.json();
      return data.translatedText || text;
    } catch (e) {
      console.error('Translation error:', e);
      return text;
    }
  }

  // Helper to get available Urdu voice name from ResponsiveVoice
  function getResponsiveUrduVoice() {
    if (!window.responsiveVoice) return null;
    const voices = window.responsiveVoice.getVoices().map(v => v.name);
    if (voices.includes('Urdu Female')) return 'Urdu Female';
    if (voices.includes('Urdu')) return 'Urdu';
    if (voices.includes('Urdu Male')) return 'Urdu Male';
    return null;
  }

  // Helper to play Urdu TTS using iSpeech API (no CORS issues, uses Audio object)
  async function playUrduTTS(text) {
    // Replace 'YOUR_API_KEY' with your actual iSpeech API key from https://www.ispeech.org/
    const apiKey = 'YOUR_API_KEY';
    const url = `https://api.ispeech.org/api/rest?apikey=${apiKey}&action=convert&text=${encodeURIComponent(text)}&voice=urdumale&format=mp3`;
    try {
      // No CORS issue: Audio object streams directly from iSpeech
      const audio = new Audio(url);
      audio.play();
      audio.onended = () => {
        setIsPlaying && setIsPlaying(false);
        setIsChatPlaying && setIsChatPlaying(false);
      };
    } catch (err) {
      alert('Urdu TTS failed: ' + err.message);
      setIsPlaying && setIsPlaying(false);
      setIsChatPlaying && setIsChatPlaying(false);
    }
  }

  // Helper to speak text in the selected language (with translation)
  async function speakSlide(text) {
    const targetLang = langMap[language] || 'en';
    let toSpeak = text;

    if (targetLang !== 'en') {
      toSpeak = await translateText(text, targetLang);
    }

    const utterance = new SpeechSynthesisUtterance(toSpeak);
    utterance.lang = language;
    const selectedVoice = getVoiceForAvatar(selectedAvatar, language);
    if (selectedVoice) {
      utterance.voice = selectedVoice;
    }
    utterance.onend = () => setIsPlaying(false);
    utterance.onerror = (event) => setIsPlaying(false);
    window.speechSynthesis.speak(utterance);
  }

  useEffect(() => {
    // When the component unmounts, stop any ongoing speech
    return () => {
      window.speechSynthesis.cancel();
    };
  }, []);

  useEffect(() => {
    if (slides && slides.length > 0) {
      // Document format: array of strings, first element is intro
      if (Array.isArray(slides) && typeof slides[0] === 'string' && slides[0].startsWith('Here is a')) {
        // Split the big string into an array of slides
        const slideArr = slides[0].split(/\nSlide \d+:/)
          .map((s, i) => i === 0 ? s : 'Slide ' + i + ':' + s)
          .filter(s => s.trim());
        setParsedSlides(parseDocumentSlides(slideArr, explanations));
      } else if (typeof slides[0] === 'string') {
        setParsedSlides(parseFirstTwoSlides(slides[0], explanations));
      } else {
        setParsedSlides(slides.slice(0, 2));
      }
      setCurrentSlideIndex(0);
    }
  }, [slides, explanations]);

  useEffect(() => {
    if (parsedSlides && parsedSlides.length > 0) {
      setCurrentSlide(parsedSlides[currentSlideIndex]);
    }
  }, [parsedSlides, currentSlideIndex]);

  useEffect(() => {
    if (isPlaying && currentSlide) {
      let textToRead = '';
      if (currentSlide.isDocument && Array.isArray(currentSlide.contentLines)) {
        textToRead = `${currentSlide.title}. ${currentSlide.contentLines.join(' ')}. ${explanations[`Slide ${currentSlideIndex + 1}`] || ""}`;
      } else {
        textToRead = `${currentSlide.title}. ${currentSlide.content}. ${explanations[`Slide ${currentSlideIndex + 1}`] || ""}`;
      }
      (async () => { await speakSlide(textToRead); })();
    }
    // eslint-disable-next-line
  }, [isPlaying, currentSlide, explanations, language, currentSlideIndex, selectedAvatar]);

  const nextSlide = () => {
    setCurrentSlideIndex((prev) => Math.min(prev + 1, parsedSlides.length - 1));
  };

  const prevSlide = () => {
    setCurrentSlideIndex((prev) => Math.max(prev - 1, 0));
  };

  const handleSlideChange = async (index) => {
    setCurrentSlideIndex(index);
    if (isPlaying) {
      window.speechSynthesis.cancel();
    }
  };

  const toggleNarration = async () => {
    window.speechSynthesis.cancel();
    if (isPlaying) {
      setIsPlaying(false);
    } else {
      let textToRead = '';
      if (currentSlide && currentSlide.explanation) {
        textToRead = currentSlide.explanation;
      } else if (currentSlide) {
        if (currentSlide.isDocument && Array.isArray(currentSlide.contentLines)) {
          textToRead = `${currentSlide.title}. ${currentSlide.contentLines.join(' ')}`;
        } else {
          textToRead = `${currentSlide.title}. ${currentSlide.content}`;
        }
      }
      if (textToRead) {
        setIsPlaying(true);
        await speakSlide(textToRead);
      }
    }
  };

  const handleQuestionSubmit = async (e) => {
    e.preventDefault();
    if (!userInput.trim()) return;

    try {
      setIsAnswering(true);
      const response = await fetch("https://fyp-chatbot-vz6d.onrender.com/qa", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${localStorage.getItem("token")}`,
        },
        body: JSON.stringify({
          question: userInput,
          context_type: "topic",
          lang_choice: language,
          context: currentSlide ? currentSlide.title : "",
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get answer");
      }

      const data = await response.json();
      setChatHistory((prev) => [
        ...prev,
        { type: "user", content: userInput },
        { type: "avatar", content: data.answer },
      ]);
      setUserInput("");
      
      // Use frontend speech synthesis for the answer
      if (language === 'ur-PK') {
        await playUrduTTS(data.answer);
        return;
      }
      const utterance = new SpeechSynthesisUtterance(data.answer);
      utterance.lang = language;
      
      const selectedVoice = getVoiceForAvatar(selectedAvatar, language);
      if (selectedVoice) {
        utterance.voice = selectedVoice;
      }
      
      utterance.onerror = (event) => {
        console.error('Chat answer speech synthesis error:', event);
      };
      
      window.speechSynthesis.speak(utterance);
    } catch (error) {
      console.error("Error getting answer:", error);
      setChatHistory((prev) => [
        ...prev,
        { type: "error", content: "Failed to get answer. Please try again." },
      ]);
    } finally {
      setIsAnswering(false);
    }
  };

  const handleStartListening = async () => {
    if (!isSpeechRecognitionSupported) {
      alert("Sorry, your browser does not support speech recognition.");
      return;
    }

    try {
      setIsListening(true);
      const recognition = new SpeechRecognition();
      recognition.lang = language;
      
      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setUserInput(transcript);
      };
      
      recognition.onend = () => {
        setIsListening(false);
      };
      
      recognition.start();
    } catch (error) {
      console.error("Error starting speech recognition:", error);
      alert(`Could not start speech recognition: ${error.message}`);
      setIsListening(false);
    }
  };

  const downloadSlides = async () => {
    if (!pptFilename) {
      alert("Could not download slides. The filename is missing.");
      return;
    }

    try {
      const response = await fetch(`https://fyp-chatbot-vz6d.onrender.com/slides/${pptFilename}`, {
        headers: {
          "Authorization": `Bearer ${localStorage.getItem("token")}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to download slides. Server responded with status ${response.status}.`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = pptFilename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Error downloading slides:", error);
      alert(`An error occurred while downloading the slides: ${error.message}`);
    }
  };

  const loadQAHistory = async () => {
    try {
      const response = await fetch("https://fyp-chatbot-vz6d.onrender.com/qa-history", {
        headers: {
          "Authorization": `Bearer ${localStorage.getItem("token")}`,
        },
      });

      if (!response.ok) {
        throw new Error("Failed to load QA history");
      }

      const data = await response.json();
      const formattedHistory = data.map((item) => ({
        type: "avatar",
        content: `Q: ${item.question}\nA: ${item.answer}`,
      }));
      setChatHistory(formattedHistory);
    } catch (error) {
      console.error("Error loading QA history:", error);
    }
  };

  const toggleLanguage = () => {
    const newLang = language === "english" ? "ur-PK" : "english";
    setLanguage(newLang);
    setSelectedLanguage(newLang);
  };

  // Utility to parse only Slide 1 and Slide 2 from slides string
  function parseFirstTwoSlides(slidesString, explanations) {
    const slideSections = slidesString.split(/Slide \d+:/).slice(1, 3); // Only first two slides
    return slideSections.map((section, idx) => {
      const lines = section.trim().split('\n');
      // Remove lines that are only === or similar
      const filteredLines = lines.filter(line => !/^=+$/.test(line.trim()));
      // Remove label prefixes
      let title = '';
      let subtitle = '';
      let image = '';
      let text = '';
      let bulletPoints = [];
      filteredLines.forEach(line => {
        if (line.startsWith('* Title:')) title = line.replace('* Title:', '').replace('"', '').trim();
        else if (line.startsWith('* Subtitle:')) subtitle = line.replace('* Subtitle:', '').replace('"', '').trim();
        else if (line.startsWith('* Image:')) image = line.replace('* Image:', '').trim();
        else if (line.startsWith('* Text:')) text = line.replace('* Text:', '').trim();
        else if (line.startsWith('* Bullet points:')) {} // skip label
        else if (line.trim().startsWith('+')) bulletPoints.push(line.replace('+', '').trim());
      });
      let content = '';
      if (subtitle) content += `<div style='font-size:1.3rem;font-weight:600;margin-bottom:0.7em;'>${subtitle}</div>`;
      if (bulletPoints.length) content += '<ul>' + bulletPoints.map(bp => `<li>${bp}</li>`).join('') + '</ul>';
      if (text) content += `<p>${text}</p>`;
      if (image) content += `<em>${image}</em>`;
      const explanation = explanations && explanations[`Slide ${idx + 1}`] ? explanations[`Slide ${idx + 1}`] : '';
      return { title, content, explanation };
    });
  }

  // Utility to parse document slides (array of strings, first element is intro)
  function parseDocumentSlides(slidesArr, explanations) {
    // Ignore the first element, map the rest
    return slidesArr.slice(1).map((slideStr, idx) => {
      const lines = slideStr.split('\n').filter(Boolean);
      // Remove lines that are only === or similar
      const filteredLines = lines.filter(line => !/^=+$/.test(line.trim()));
      const title = filteredLines[0] || `Slide ${idx + 1}`;
      const contentLines = filteredLines.slice(1);
      const explanation = explanations && explanations[`Slide ${idx + 1}`] ? explanations[`Slide ${idx + 1}`] : '';
      return { title, contentLines, explanation, isDocument: true };
    });
  }

  // Utility to get the correct avatar video path
  function getAvatarVideo(selectedAvatar) {
    return selectedAvatar === 0 ? "/assets/boyavatar.mp4" : "/assets/girlavatar.mp4";
  }

  // Utility to get the correct avatar image path
  function getAvatarImage(selectedAvatar) {
    return selectedAvatar === 0 ? "/assets/boy.jpg" : "/assets/girl.jpg";
  }

  // Utility to get a matching voice for the avatar
  function getVoiceForAvatar(avatar, langCode) {
    const voices = window.speechSynthesis.getVoices();
    console.log('Available voices:', voices.map(v => `${v.name} (${v.lang})`));
    
    // Filter voices by language code
    let filtered = voices.filter(v => v.lang === langCode);
    
    // If no exact match, try language family matching
    if (filtered.length === 0) {
      const langFamily = langCode.split('-')[0];
      filtered = voices.filter(v => v.lang.startsWith(langFamily));
    }
    
    function matchVoice(voice, keywords) {
      const name = voice.name.toLowerCase();
      const uri = (voice.voiceURI || '').toLowerCase();
      return keywords.some(k => name.includes(k) || uri.includes(k));
    }
    
    let voice = null;
    if (avatar === 0) { // Male avatar
      // Male voice keywords for different languages
      const maleKeywords = {
        'en-US': ['male', 'man', 'boy', 'david', 'mark', 'alex', 'tom', 'mike'],
        'es-ES': ['male', 'hombre', 'carlos', 'juan', 'pedro', 'miguel'],
        'de-DE': ['male', 'mann', 'hans', 'klaus', 'wolfgang', 'thomas'],
        'fr-FR': ['male', 'homme', 'pierre', 'jean', 'paul', 'michel']
      };
      
      const keywords = maleKeywords[langCode] || maleKeywords['en-US'];
      voice = filtered.find(v => matchVoice(v, keywords))
        || filtered.find(v => v.gender === "male")
        || filtered[0];
    } else { // Female avatar
      // Female voice keywords for different languages
      const femaleKeywords = {
        'en-US': ['female', 'woman', 'girl', 'zira', 'eva', 'susan', 'samantha', 'linda', 'victoria'],
        'es-ES': ['female', 'mujer', 'maria', 'ana', 'carmen', 'lucia', 'isabel'],
        'de-DE': ['female', 'frau', 'anna', 'maria', 'greta', 'helena', 'sophie'],
        'fr-FR': ['female', 'femme', 'marie', 'anne', 'sophie', 'claire', 'julie']
      };
      
      const keywords = femaleKeywords[langCode] || femaleKeywords['en-US'];
      voice = filtered.find(v => matchVoice(v, keywords))
        || filtered.find(v => v.gender === "female")
        || filtered[0];
    }
    
    if (voice) {
      console.log('Selected voice for avatar', avatar === 0 ? 'male' : 'female', ':', voice.name, voice.lang, voice.gender);
    } else {
      console.warn('No matching voice found for avatar', avatar, 'language:', langCode);
      console.log('Available voices for language', langCode, ':', filtered);
    }
    return voice;
  }

  // Load voices when component mounts
  useEffect(() => {
    const loadVoices = () => {
      const voices = window.speechSynthesis.getVoices();
      console.log('Loaded voices:', voices.length);
      if (voices.length > 0) {
        console.log('Available languages:', [...new Set(voices.map(v => v.lang))]);
      }
    };
    
    // Load voices immediately if available
    loadVoices();
    
    // Also listen for voices to be loaded
    window.speechSynthesis.onvoiceschanged = loadVoices;
    
    return () => {
      window.speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  // Update Sidebar to accept onNewChat and onHistory props
  function Sidebar({ onNewChat, onHistory, expanded, onToggle }) {
    const navigate = useNavigate();
    return (
      <div
        className="sidebar"
        style={{
          width: expanded ? 250 : 60,
          minHeight: '100vh',
          background: '#fff',
          boxShadow: '2px 0 8px rgba(0,0,0,0.04)',
          padding: expanded ? '1.5rem 0' : '1.5rem 0.5rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.5rem',
          alignItems: expanded ? 'flex-start' : 'center',
          transition: 'width 0.2s',
          zIndex: 30
        }}
      >
        <button
          onClick={onToggle}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            marginBottom: 16,
            fontSize: 24,
            alignSelf: expanded ? 'flex-end' : 'center',
          }}
          title={expanded ? 'Collapse sidebar' : 'Expand sidebar'}
        >
          <FaBars />
        </button>
        {expanded && (
          <div style={{ fontWeight: 700, fontSize: 18, marginBottom: 16, textAlign: 'center', width: '100%' }}>Menu</div>
        )}
        <div style={{ padding: expanded ? '0 1.5rem' : 0, width: '100%' }}>
          <div style={{ margin: '1rem 0', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 10 }} onClick={onNewChat}>
            <span role="img" aria-label="chat">🗨️</span>
            {expanded && 'New Chat'}
          </div>
          <div style={{ margin: '1rem 0', display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }} onClick={onHistory}>
            <span role="img" aria-label="history">⏳</span>
            {expanded && 'History'}
          </div>
          <div style={{ margin: '1rem 0', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span role="img" aria-label="saved">🔖</span>
            {expanded && 'Saved Chats'}
          </div>
          <div style={{ margin: '1rem 0', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span role="img" aria-label="profile">👤</span>
            {expanded && 'My Profile'}
          </div>
          <div style={{ margin: '1rem 0', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span role="img" aria-label="courses">📚</span>
            {expanded && 'My Courses'}
          </div>
          <div style={{ margin: '1rem 0', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span role="img" aria-label="tips">💡</span>
            {expanded && 'Learning Tips'}
          </div>
          <div style={{ margin: '1rem 0', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span role="img" aria-label="progress">��</span>
            {expanded && 'Progress Tracker'}
          </div>
          <div style={{ margin: '1rem 0', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span role="img" aria-label="ai-tutor">🤖</span>
            {expanded && 'AI Tutor Settings'}
          </div>
          <div style={{ margin: '1rem 0', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span role="img" aria-label="settings">⚙️</span>
            {expanded && 'Settings'}
          </div>
        </div>
      </div>
    );
  }

  // Add these handlers for chat play/stop
  const handlePlayChat = async () => {
    let textToSpeak = userInput;
    if (!textToSpeak) {
      const lastAvatarMsg = [...chatHistory].reverse().find(msg => msg.type === 'avatar');
      if (lastAvatarMsg) textToSpeak = lastAvatarMsg.content;
    }
    if (!textToSpeak) return;

    if (language === 'ur-PK') {
      await playUrduTTS(textToSpeak);
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new window.SpeechSynthesisUtterance(textToSpeak);
    utterance.lang = language;
    const selectedVoice = getVoiceForAvatar(selectedAvatar, language);
    if (selectedVoice) {
      utterance.voice = selectedVoice;
    }
    utterance.onend = () => setIsChatPlaying(false);
    utterance.onerror = (event) => setIsChatPlaying(false);
    setIsChatPlaying(true);
    window.speechSynthesis.speak(utterance);
  };

  const handleStopChat = () => {
    window.speechSynthesis.cancel();
    setIsChatPlaying(false);
  };

  useEffect(() => {
    // Collapse sidebar when processing page is shown
    if (processingComplete) {
      setSidebarExpanded(false);
    }
  }, [processingComplete]);

  useEffect(() => {
    if (showChat) {
      loadQAHistory();
    }
  }, [showChat]);

  // Palette of gradients for slides
  const slideBackgrounds = [
    'radial-gradient(ellipse at 60% 40%, #232946 60%, #181c2a 100%)',
    'radial-gradient(ellipse at 60% 40%, #2563eb 60%, #7c3aed 100%)',
    'radial-gradient(ellipse at 60% 40%, #00cfff 60%, #e946fd 100%)',
    'radial-gradient(ellipse at 60% 40%, #43e97b 60%, #38f9d7 100%)',
    'radial-gradient(ellipse at 60% 40%, #fa709a 60%, #fee140 100%)',
    'radial-gradient(ellipse at 60% 40%, #f093fb 60%, #f5576c 100%)',
  ];

  function parseQuizMarkdown(quizContent) {
    // Split by '### Question' and filter out empty
    const questions = quizContent.split(/### Question \d+/).slice(1);
    return questions.map(qBlock => {
      // Extract question text (up to first \n)
      const [questionLine, ...rest] = qBlock.trim().split('\n');
      // Find all options (A), B), C), D))
      const options = [];
      let correct = '';
      rest.forEach(line => {
        const optMatch = line.match(/^([A-D])\)\s(.+)/);
        if (optMatch) {
          options.push(optMatch[2]);
        }
        const correctMatch = line.match(/^Correct answer: ([A-D])\)/);
        if (correctMatch) {
          correct = correctMatch[1].toLowerCase();
        }
      });
      return {
        question: questionLine.trim(),
        options,
        correct, // 'a', 'b', etc.
      };
    });
  }

  const fetchQuiz = async () => {
    setQuizLoading(true);
    setShowQuiz(true);
    setQuizSubmitted(false);
    setQuizScore(0);
    try {
      const response = await fetch("https://fyp-chatbot-vz6d.onrender.com/generate-quiz", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${localStorage.getItem("token")}`,
        },
        body: JSON.stringify({
          slide_filename: pptFilename,
          lang_choice: language,
        }),
      });
      if (!response.ok) throw new Error("Failed to fetch quiz");
      const data = await response.json();
      // Parse the markdown quiz content
      const parsed = parseQuizMarkdown(data.quiz_content || "");
      setQuizQuestions(parsed);
    } catch (err) {
      alert("Could not load quiz: " + err.message);
      setQuizQuestions([]);
    }
    setQuizLoading(false);
  };

  console.log('currentSlideIndex', currentSlideIndex, 'parsedSlides.length', parsedSlides.length);

  if (!processingComplete) {
    return (
      <>
        <h2>Processing Your Request</h2>
        <p>Our AI is analyzing your content and preparing your personalized learning experience</p>
        <div className="loading-animation">
          <div className="spinner"></div>
        </div>
        <p className="processing-message">This will just take a moment...</p>
      </>
    );
  }

  return (
    <div className="processing-root" style={{ display: 'flex', minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
      <Sidebar
        onNewChat={onNewChat}
        onHistory={() => setShowHistory(true)}
        expanded={sidebarExpanded}
        onToggle={() => setSidebarExpanded((prev) => !prev)}
      />
      {showHistory && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'rgba(0,0,0,0.35)',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <HistoryPage show={true} onClose={() => setShowHistory(false)} />
        </div>
      )}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: '100vh', position: 'relative' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, margin: '24px 0 0 32px' }}>
          <label htmlFor="language-select" style={{ fontWeight: 500 }}>Avatar Language:</label>
          <select
            id="language-select"
            value={language}
            onChange={e => {
              setLanguage(e.target.value);
              setSelectedLanguage(e.target.value);
            }}
            style={{
              padding: '6px 12px',
              borderRadius: 8,
              border: '1px solid #e3eafc',
              fontSize: '1rem',
              background: '#f8fafd',
              color: '#333'
            }}
          >
            <option value="en-US">English</option>
            <option value="es-ES">Spanish</option>
            <option value="de-DE">German</option>
            <option value="fr-FR">French</option>
          </select>
        </div>
        <div className="ppt-layout" style={{ flex: 1, display: 'flex', alignItems: 'stretch', justifyContent: 'center', padding: '2rem 0 0 0' }}>
          <div className="ppt-sidebar" style={{
            width: 260,
            background: 'transparent',
            borderRadius: 18,
            boxShadow: 'none',
            padding: '1.5rem 0.5rem',
            marginRight: 32,
            display: 'flex',
            flexDirection: 'column',
            gap: '1rem',
            overflowY: 'auto',
            maxHeight: '80vh',
            height: '100%',
            border: 'none',
          }}>
            {parsedSlides.map((slide, idx) => {
              // Clean up subtitle if it starts with 'Title: "* Title:'
              let subtitle = '';
              if (slide.contentLines && slide.contentLines[0]) {
                subtitle = slide.contentLines[0].replace(/^\u0002A?\s*Title:\s*"*\s*Title:\s*/i, '').replace(/^\u00002A?\s*Title:\s*/i, '').replace(/^"/, '').replace(/"$/, '');
              }
              return (
                <div
                  key={idx}
                  className={`ppt-thumbnail${idx === currentSlideIndex ? ' active' : ''}`}
                  onClick={() => setCurrentSlideIndex(idx)}
                  style={{
                    background: slideBackgrounds[0],
                    borderRadius: 18,
                    boxShadow: idx === currentSlideIndex ? '0 4px 16px #2563eb33' : '0 2px 8px #e3eafc',
                    border: idx === currentSlideIndex ? '2px solid #2563eb' : '2px solid #e3eafc',
                    padding: '3.5rem 2rem 3.5rem 2rem',
                    minHeight: 180,
                    cursor: 'pointer',
                    transition: 'all 0.18s',
                    fontWeight: 500,
                    color: '#fff',
                    fontSize: 15,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 6,
                    position: 'relative',
                    alignItems: 'flex-start',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 2 }}>
                    <span style={{
                      fontWeight: 700,
                      fontSize: 22,
                      color: '#fff',
                      background: 'rgba(255,255,255,0.08)',
                      borderRadius: '50%',
                      width: 36,
                      height: 36,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      border: idx === currentSlideIndex ? '2.5px solid #2563eb' : '2.5px solid #e3eafc',
                      boxShadow: idx === currentSlideIndex ? '0 2px 8px #2563eb22' : 'none',
                    }}>{idx + 1}</span>
                    <span style={{
                      fontWeight: 700,
                      fontSize: 17,
                      color: '#fff',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      maxWidth: 140,
                    }}>{`Slide ${idx + 1}: ${slide.title.slice(0, 18)}${slide.title.length > 18 ? '...' : ''}`}</span>
                  </div>
                  {subtitle && (
                    <div style={{ fontSize: 14, color: '#fff', fontWeight: 400, marginLeft: 50, marginTop: 0 }}>
                      • {subtitle.slice(0, 40)}{subtitle.length > 40 ? '...' : ''}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          <div className="ppt-main" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div className="slide-content" style={{ width: '100%' }}>
              {currentSlide ? (
                <div className="ppt-slide" style={{
                  aspectRatio: '16/9',
                  minWidth: '700px',
                  minHeight: '400px',
                  maxWidth: '1100px',
                  maxHeight: '620px',
                  margin: '0 auto',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'flex-start',
                  alignItems: 'flex-start',
                  position: 'relative',
                  background: slideBackgrounds[0],
                  borderRadius: 24,
                  boxShadow: '0 4px 32px 0 #181c2a, 0 0 0 1px #232946',
                  padding: '3.5rem 3.5rem 2.5rem 3.5rem',
                  overflow: 'hidden',
                }}>
                  <h2 className="ppt-title" style={{
                    fontSize: '3.2rem',
                    fontWeight: 800,
                    marginBottom: '2.2rem',
                    color: '#fff',
                    textAlign: 'left',
                    lineHeight: 1.1,
                    letterSpacing: '-1px',
                    textShadow: '0 4px 32px #181c2a',
                  }}>{currentSlide.title}</h2>
                  {currentSlide.isDocument && Array.isArray(currentSlide.contentLines) ? (
                    <ul className="ppt-bullets" style={{ fontSize: 22, color: '#e0e6f7', marginLeft: 0, marginBottom: 18, fontWeight: 400, listStyle: 'none', padding: 0 }}>
                      {currentSlide.contentLines.map((line, idx) => (
                        <li key={idx} style={{ marginBottom: 18 }}>{line.trim()}</li>
                      ))}
                    </ul>
                  ) : (
                    <div
                      className="ppt-main-content"
                      style={{ fontSize: 22, color: '#e0e6f7', marginBottom: 18, marginLeft: 0, fontWeight: 400 }}
                      dangerouslySetInnerHTML={{ __html: currentSlide.content }}
                    />
                  )}
                  {/* Neon/gradient glow effect at the bottom */}
                  <div style={{
                    position: 'absolute',
                    left: 0,
                    bottom: 0,
                    width: '100%',
                    height: 120,
                    background: 'radial-gradient(ellipse at 30% 80%, #00cfff88 0%, transparent 70%), radial-gradient(ellipse at 70% 90%, #e946fd88 0%, transparent 70%)',
                    zIndex: 1,
                    pointerEvents: 'none',
                  }} />
                  <video
                    src={getAvatarVideo(selectedAvatar)}
                    className={`avatar-bottom-right${isPlaying ? ' speaking' : ''}`}
                    autoPlay
                    loop
                    muted
                    playsInline
                  />
                  <div className="slide-navigation-bar" style={{ marginTop: 'auto', display: 'flex', gap: 12, justifyContent: 'center', alignItems: 'center' }}>
                    {parsedSlides && parsedSlides.length > 1 && (
                      <>
                        <button
                          onClick={prevSlide}
                          disabled={currentSlideIndex === 0}
                          className="nav-button"
                        >
                          Previous
                        </button>
                        <span className="slide-counter" style={{ alignSelf: 'center' }}>
                          Slide {currentSlideIndex + 1} of {parsedSlides.length}
                        </span>
                        <button
                          onClick={nextSlide}
                          disabled={currentSlideIndex === parsedSlides.length - 1}
                          className="nav-button"
                        >
                          Next
                        </button>
                      </>
                    )}
                    <button onClick={toggleNarration} className="nav-button">
                      {isPlaying ? "Stop" : "Play Explanation"}
                    </button>
                    <button onClick={downloadSlides} className="download-button">
                      Download Slides
                    </button>
                  </div>
                </div>
              ) : (
                <p>No slide available</p>
              )}
            </div>
          </div>
        </div>
        {/* Floating Chatbot Icon */}
        {!showChat && (
          <button
            className="floating-chatbot-btn"
            onClick={() => setShowChat(true)}
            title="Open Chat"
            style={{
              position: 'fixed',
              bottom: 40,
              right: 40,
              zIndex: 120,
              background: '#2563eb',
              color: '#fff',
              border: 'none',
              borderRadius: '50%',
              width: 60,
              height: 60,
              boxShadow: '0 4px 16px rgba(0,0,0,0.18)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 32,
              cursor: 'pointer',
              transition: 'background 0.2s',
            }}
          >
            <FaComments />
          </button>
        )}

        {/* Chat Modal */}
        {showChat && (
          <div className="chat-modal" style={{
            position: 'fixed',
            bottom: 32,
            right: 32,
            width: 540,
            maxWidth: '98vw',
            height: 540,
            maxHeight: '98vh',
            background: '#fff',
            borderRadius: 22,
            boxShadow: '0 8px 40px rgba(37,99,235,0.13), 0 1.5px 8px rgba(0,0,0,0.04)',
            border: '1.5px solid #e3eafc',
            display: 'flex',
            flexDirection: 'column',
            zIndex: 200,
            animation: 'fadeIn 0.3s',
            overflow: 'hidden',
            padding: 0,
          }}>
            {/* Modal Close Button */}
            <button
              className="chat-close-btn"
              style={{
                position: 'absolute',
                top: 18,
                right: 18,
                background: '#fff',
                border: 'none',
                borderRadius: '50%',
                width: 38,
                height: 38,
                boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 201,
                cursor: 'pointer',
                transition: 'box-shadow 0.2s, background 0.2s, transform 0.15s',
              }}
              onClick={() => setShowChat(false)}
              title="Close Chat"
            >
              <FaTimes size={18} color="#333" />
            </button>
            {/* Chat history */}
            <div className="chat-history" style={{ flex: 1, overflowY: 'auto', padding: '32px 32px 24px 32px', background: 'linear-gradient(135deg, #f7fafd 60%, #e3eafc 100%)', borderBottom: '1px solid #e3eafc', maxHeight: 600 }}>
              {chatHistory.length === 0 ? (
                <div
                  style={{
                    height: '100%',
                    minHeight: 320,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#7c3aed',
                    background: 'linear-gradient(135deg, #f7fafd 60%, #e3eafc 100%)',
                    borderRadius: 18,
                    boxShadow: '0 2px 16px #e3eafc',
                    margin: 24,
                    padding: 32,
                    textAlign: 'center',
                    animation: 'fadeIn 0.5s',
                  }}
                >
                  <div style={{ fontSize: 64, marginBottom: 16 }}>🤖</div>
                  <div style={{ fontWeight: 700, fontSize: 24, marginBottom: 8 }}>
                    Welcome to your AI Chat!
                  </div>
                  <div style={{ fontSize: 16, color: '#6b7280', marginBottom: 18 }}>
                    Ask me anything about your slides, or try one of these:
                  </div>
                  <div style={{
                    background: '#ede9fe',
                    color: '#7c3aed',
                    borderRadius: 12,
                    padding: '10px 18px',
                    fontSize: 15,
                    fontWeight: 500,
                    marginBottom: 6,
                    display: 'inline-block'
                  }}>
                    "Summarize slide 1 for me"
                  </div>
                  <div style={{
                    background: '#ede9fe',
                    color: '#7c3aed',
                    borderRadius: 12,
                    padding: '10px 18px',
                    fontSize: 15,
                    fontWeight: 500,
                    display: 'inline-block'
                  }}>
                    "What is data normalization?"
                  </div>
                </div>
              ) : (
                chatHistory.map((message, index) => (
                <div key={index} className={`chat-message ${message.type}`} style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 8 }}>
                  {message.type === "avatar" && (
                    <video
                      src={getAvatarVideo(selectedAvatar)}
                      className="avatar-image"
                      autoPlay
                      loop
                      muted
                      playsInline
                      style={{ width: 40, height: 40, borderRadius: "50%", objectFit: "cover" }}
                    />
                  )}
                  <div className="message-content" style={{ background: message.type === 'avatar' ? '#eaf1fb' : '#f3f3f3', borderRadius: 12, padding: '8px 16px', maxWidth: 600 }}>{message.content}</div>
                </div>
                ))
              )}
            </div>
            {/* Input bar (pill-shaped, minimal, modern) */}
            <div className="chatgpt-input-bar pill" style={{ background: '#f8fafd', borderRadius: 32, padding: '10px 18px', boxShadow: '0 2px 8px rgba(0,0,0,0.03)', margin: '18px 24px 18px 24px', display: 'flex', alignItems: 'center', gap: 10 }}>
              <button className="attach-btn" title="Attach file" style={{ background: 'none', border: 'none', color: '#888', fontSize: '1.3rem', cursor: 'pointer', borderRadius: '50%', padding: '7px', transition: 'background 0.2s' }}>
                <FaPaperclip />
              </button>
              {/* Play/Stop button for avatar speech */}
              {isChatPlaying ? (
                <button
                  type="button"
                  onClick={handleStopChat}
                  className="playstop-btn"
                  style={{ background: '#f8fafd', border: 'none', color: '#7c3aed', borderRadius: '50%', width: 38, height: 38, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.3rem', marginRight: 2, cursor: 'pointer', transition: 'background 0.2s' }}
                  title="Stop Avatar Speech"
                >
                  <FaPause />
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handlePlayChat}
                  className="playstop-btn"
                  style={{ background: '#f8fafd', border: 'none', color: '#7c3aed', borderRadius: '50%', width: 38, height: 38, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.3rem', marginRight: 2, cursor: 'pointer', transition: 'background 0.2s' }}
                  title="Play Avatar Speech"
                >
                  <FaPlay />
                </button>
              )}
              <input
                type="text"
                value={userInput}
                onChange={e => setUserInput(e.target.value)}
                placeholder="Type your message here..."
                className="chatgpt-input"
                disabled={isAnswering}
                style={{ flex: 1, minWidth: 0, background: 'transparent', border: 'none', fontSize: '1.08rem', padding: '12px 0', outline: 'none' }}
              />
              <button
                type="button"
                onClick={handleStartListening}
                className={`mic-button${isListening ? ' listening' : ''}`}
                disabled={isAnswering || !isSpeechRecognitionSupported}
                style={{
                  background: '#f8fafd',
                  border: 'none',
                  color: '#7c3aed',
                  borderRadius: '50%',
                  width: 38,
                  height: 38,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '1.3rem',
                  marginRight: 2,
                  cursor: isAnswering || !isSpeechRecognitionSupported ? 'not-allowed' : 'pointer',
                  transition: 'background 0.2s'
                }}
                title={
                  !isSpeechRecognitionSupported
                    ? "Speech recognition not supported in your browser"
                    : "Start listening"
                }
              >
                <FaMicrophone />
              </button>
              <button
                type="button"
                onClick={handleQuestionSubmit}
                className="send-button"
                style={{ background: '#7c3aed', color: '#fff', borderRadius: '0 22px 22px 0', fontSize: '1.4rem', width: 56, height: 48, display: 'flex', alignItems: 'center', justifyContent: 'center', border: 'none', marginLeft: 4 }}
                disabled={isAnswering || !userInput.trim()}
                title="Send"
              >
                <FaPaperPlane />
              </button>
            </div>
          </div>
        )}
      </div>
      {showQuiz && (
        <div style={{
          position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh',
          background: 'rgba(0,0,0,0.35)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>
          <div style={{
            background: '#fff', borderRadius: 18, padding: 0, minWidth: 400, maxWidth: 600, boxShadow: '0 8px 40px #23294633', position: 'relative', display: 'flex', flexDirection: 'column', maxHeight: '90vh', width: '100%'
          }}>
            {/* Sticky header */}
            <div style={{ position: 'sticky', top: 0, background: '#fff', zIndex: 2, borderTopLeftRadius: 18, borderTopRightRadius: 18, padding: '24px 32px 12px 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid #e3eafc' }}>
              <h2 style={{ margin: 0, fontSize: 26, fontWeight: 800 }}>Quiz</h2>
              <button
                onClick={() => setShowQuiz(false)}
                style={{ background: 'none', border: 'none', fontSize: 28, cursor: 'pointer', color: '#7c3aed', fontWeight: 700 }}
                title="Close Quiz"
              >×</button>
            </div>
            {/* Scrollable questions area */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '18px 32px 120px 32px' }}>
              {quizLoading ? (
                <div>Loading quiz...</div>
              ) : quizQuestions.length === 0 ? (
                <div>No quiz available.</div>
              ) : !quizSubmitted ? (
                <form onSubmit={e => {
                  e.preventDefault();
                  let score = 0;
                  quizQuestions.forEach((q, idx) => {
                    if (quizAnswers[idx] === q.correct) score++;
                  });
                  setQuizScore(score);
                  setQuizSubmitted(true);
                }}>
                  {quizQuestions.map((q, idx) => (
                    <div key={idx} style={{ marginBottom: 28, padding: 18, borderRadius: 12, background: '#f8fafd', boxShadow: '0 1px 4px #e3eafc' }}>
                      <div style={{ fontWeight: 700, fontSize: 18, marginBottom: 10 }}>{idx + 1}. {q.question}</div>
                      {q.options.map((opt, oidx) => (
                        <label key={oidx} style={{ display: 'block', marginBottom: 7, fontSize: 16, cursor: 'pointer', paddingLeft: 6 }}>
                          <input
                            type="radio"
                            name={`quiz-q${idx}`}
                            value={String.fromCharCode(97 + oidx)}
                            checked={quizAnswers[idx] === String.fromCharCode(97 + oidx)}
                            onChange={() => setQuizAnswers(a => ({ ...a, [idx]: String.fromCharCode(97 + oidx) }))}
                            required
                            style={{ marginRight: 8 }}
                          /> {opt}
                        </label>
                      ))}
                    </div>
                  ))}
                  {/* Fixed submit button */}
                  <div style={{
                    position: 'fixed',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    bottom: 40,
                    width: 'calc(100% - 64px)',
                    maxWidth: 600,
                    background: '#fff',
                    padding: '18px 0 0 0',
                    zIndex: 3000,
                    display: 'flex',
                    justifyContent: 'center',
                    borderBottomLeftRadius: 18,
                    borderBottomRightRadius: 18,
                    boxShadow: '0 2px 8px #2563eb22',
                  }}>
                    <button type="submit" style={{
                      background: '#2563eb', color: '#fff', border: 'none', borderRadius: 8, padding: '12px 36px', fontSize: 18, fontWeight: 700, cursor: 'pointer', boxShadow: '0 2px 8px #2563eb22', letterSpacing: '1px'
                    }}>Submit Quiz</button>
                  </div>
                </form>
              ) : (
                <div>
                  <div style={{ fontWeight: 700, fontSize: 22, marginBottom: 18, textAlign: 'center' }}>
                    Your Score: {quizScore} / {quizQuestions.length}
                  </div>
                  {quizQuestions.map((q, idx) => {
                    const userAns = quizAnswers[idx];
                    const isCorrect = userAns === q.correct;
                    return (
                      <div key={idx} style={{ marginBottom: 24, padding: 18, borderRadius: 12, background: isCorrect ? '#e6fbe8' : '#fbeaea', boxShadow: '0 1px 4px #e3eafc' }}>
                        <div style={{ fontWeight: 700, fontSize: 18, marginBottom: 10 }}>{idx + 1}. {q.question}</div>
                        {q.options.map((opt, oidx) => {
                          const optKey = String.fromCharCode(97 + oidx);
                          const isUser = userAns === optKey;
                          const isRight = q.correct === optKey;
                          return (
                            <div key={oidx} style={{
                              display: 'flex', alignItems: 'center', marginBottom: 7, fontSize: 16,
                              color: isRight ? '#15803d' : isUser && !isRight ? '#b91c1c' : '#333',
                              fontWeight: isRight || isUser ? 700 : 400,
                              background: isUser ? (isRight ? '#bbf7d0' : '#fee2e2') : 'transparent',
                              borderRadius: 6, paddingLeft: 6, paddingRight: 6
                            }}>
                              <input
                                type="radio"
                                checked={isUser}
                                readOnly
                                style={{ marginRight: 8 }}
                              />
                              {opt}
                              {isRight && <span style={{ marginLeft: 8, color: '#15803d', fontWeight: 800 }}>&#10003;</span>}
                              {isUser && !isRight && <span style={{ marginLeft: 8, color: '#b91c1c', fontWeight: 800 }}>&#10007;</span>}
                            </div>
                          );
                        })}
                        {!isCorrect && (
                          <div style={{ color: '#b91c1c', fontWeight: 600, marginTop: 6 }}>
                            Correct answer: <span style={{ color: '#15803d' }}>{q.options[q.correct.charCodeAt(0) - 97]}</span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                  <div style={{ display: 'flex', justifyContent: 'center', marginTop: 18 }}>
                    <button onClick={() => setShowQuiz(false)} style={{
                      background: '#43e97b', color: '#fff', border: 'none', borderRadius: 8, padding: '12px 36px', fontSize: 18, fontWeight: 700, cursor: 'pointer', boxShadow: '0 2px 8px #43e97b22', letterSpacing: '1px'
                    }}>Close</button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      {/* Take Quiz button fixed at bottom center on last slide */}
      {currentSlideIndex === parsedSlides.length - 1 && (
        <div style={{
          position: 'fixed',
          left: 0,
          bottom: 24,
          width: '100vw',
          display: 'flex',
          justifyContent: 'center',
          zIndex: 150,
          pointerEvents: 'none'
        }}>
          <button
            onClick={fetchQuiz}
            className="quiz-button"
            style={{
              background: '#7c3aed',
              color: '#fff',
              border: 'none',
              borderRadius: 16,
              padding: '18px 48px',
              fontSize: 22,
              fontWeight: 800,
              cursor: 'pointer',
              boxShadow: '0 2px 16px #7c3aed44',
              transition: 'background 0.2s',
              letterSpacing: '1px',
              pointerEvents: 'auto'
            }}
          >
            Take Quiz
          </button>
        </div>
      )}
    </div>
  );
};

export default ProcessingView;