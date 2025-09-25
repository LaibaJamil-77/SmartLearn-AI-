"use client"
/* eslint-disable no-undef */

import { useState, useEffect } from "react"
import {
  FaFacebook,
  FaInstagram,
  FaTwitter,
  FaYoutube,
  FaEnvelope,
  FaCloudUploadAlt,
  FaKeyboard,
  FaArrowRight,
  FaGoogle,
  FaBars,
  FaComments,
  FaHistory,
  FaBookmark,
  FaCog,
  FaQuestionCircle,
  FaTimes,
  FaDownload,
  FaUser,
  FaChalkboardTeacher,
  FaLightbulb,
  FaGraduationCap,
  FaRobot,
  FaPlay,
  FaPause,
  FaChevronLeft,
  FaChevronRight,
} from "react-icons/fa"

import { motion } from "framer-motion"
import LandingPage from "./LandingPage"
import LoginForm from "./LoginForm"
import SignupForm from "./SignupForm"
import AvatarSelection from "./AvatarSelection"
import InputSelection from "./InputSelection"
import ProcessingView from "./ProcessingView"
import "./header.css"

const Head = () => {
  // Main state for different views
  const [currentView, setCurrentView] = useState("landing") // "landing", "login", "dashboard"

  // Login/Signup state
  const [isLogin, setIsLogin] = useState(true)

  // Dashboard state
  const [step, setStep] = useState(1)
  const [selectedAvatar, setSelectedAvatar] = useState(null)
  const [inputMethod, setInputMethod] = useState(null)
  const [topic, setTopic] = useState("")
  const [instructions, setInstructions] = useState("")
  const [numSlides, setNumSlides] = useState(10)
  const [file, setFile] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [activeSlide, setActiveSlide] = useState(0)
  const [processingComplete, setProcessingComplete] = useState(false)
  const [videoPlayed, setVideoPlayed] = useState(false)
  const [userInput, setUserInput] = useState("")
  const [showLanguageSelector, setShowLanguageSelector] = useState(false)
  const [selectedLanguage, setSelectedLanguage] = useState("en")
  const [slides, setSlides] = useState([])
  const [pptFilename, setPptFilename] = useState("")
  const [explanations, setExplanations] = useState({})
  const [showProcessingView, setShowProcessingView] = useState(false)

  const handlePlay = () => {
    console.log("Play clicked")
  }

  const handlePause = () => {
    console.log("Pause clicked")
  }

  const handleResume = () => {
    console.log("Resume clicked")
  }

  const handleDownload = () => {
    const blob = new Blob([userInput], { type: "text/plain;charset=utf-8" })
    const link = document.createElement("a")
    link.href = URL.createObjectURL(blob)
    link.download = "message.txt"
    link.click()
  }

  const startListening = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition

    if (!SpeechRecognition) {
      alert("Speech recognition not supported in this browser.")
      return
    }

    const recognition = new SpeechRecognition()
    recognition.lang = "en-US"
    recognition.interimResults = false
    recognition.maxAlternatives = 1

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript
      setUserInput(transcript)
    }

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error)
    }

    recognition.start()
  }

  const toggleForm = () => {
    setIsLogin(!isLogin)
  }

  const handleAvatarSelect = (index) => {
    setSelectedAvatar(index)
  }

  const handleContinue = async () => {
    if (step === 1 && selectedAvatar !== null) {
      setStep(2)
      return
    }

    if (step === 2 && inputMethod) {
      setStep(3) // Move to processing screen immediately

      try {
        let response
        const token = localStorage.getItem("token")

        if (inputMethod === "upload" && file) {
          const formData = new FormData()
          formData.append("file", file)
          response = await fetch("https://fyp-chatbot-vz6d.onrender.com/generate-slides/document", {
            method: "POST",
            headers: {
              Authorization: `Bearer ${token}`,
            },
            body: formData,
          })
        } else if (inputMethod === "text" && topic) {
          response = await fetch("https://fyp-chatbot-vz6d.onrender.com/generate-slides/topic", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({
              topic,
              instructions,
              num_slides: numSlides,
              lang_choice: selectedLanguage,
            }),
          })
        }

        if (!response || !response.ok) {
          const errorData = response ? await response.json() : { detail: "An unknown error occurred" }
          throw new Error(errorData.detail || "Failed to generate slides")
        }

        const result = await response.json()
        
        setSlides(result.slides)
        setExplanations(result.explanations)
        setPptFilename(result.ppt_filename)
        setProcessingComplete(true)

      } catch (error) {
        console.error("Error generating slides:", error)
        alert(`Error: ${error.message}`)
        setStep(2) // Go back on error
      }
    }
  }

  const handleVideoEnd = () => {
    setVideoPlayed(true)
  }

  const renderView = () => {
    if (currentView === "login") {
      return (
        <div className="centered-auth-container">
          {isLogin ? (
            <LoginForm setCurrentView={setCurrentView} toggleForm={toggleForm} />
          ) : (
            <SignupForm setCurrentView={setCurrentView} toggleForm={toggleForm} />
          )}
        </div>
      )
    } else if (currentView === "dashboard") {
      return (
        <div className="dashboard-container">
          <div className="dashboard-header">
            <h1>SmartLearn AI</h1>
            <p>Your Intelligent Virtual Tutor</p>
          </div>

          <motion.div
            className="dashboard-content"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {step === 1 && (
              <AvatarSelection
                selectedAvatar={selectedAvatar}
                handleAvatarSelect={handleAvatarSelect}
                handleContinue={handleContinue}
              />
            )}

            {step === 2 && (
              <InputSelection
                sidebarOpen={sidebarOpen}
                toggleSidebar={() => setSidebarOpen(!sidebarOpen)}
                inputMethod={inputMethod}
                setInputMethod={setInputMethod}
                file={file}
                setFile={setFile}
                topic={topic}
                setTopic={setTopic}
                handleContinue={handleContinue}
              />
            )}

            {step === 3 && (
              <ProcessingView
                processingComplete={processingComplete}
                videoPlayed={videoPlayed}
                handleVideoEnd={handleVideoEnd}
                slides={slides}
                pptFilename={pptFilename}
                explanations={explanations}
                activeSlide={activeSlide}
                setActiveSlide={setActiveSlide}
                handlePlay={handlePlay}
                handlePause={handlePause}
                handleResume={handleResume}
                handleDownload={handleDownload}
                userInput={userInput}
                setUserInput={setUserInput}
                startListening={startListening}
                showLanguageSelector={showLanguageSelector}
                setShowLanguageSelector={setShowLanguageSelector}
                selectedLanguage={selectedLanguage}
                setSelectedLanguage={setSelectedLanguage}
                selectedAvatar={selectedAvatar}
                onNewChat={() => {
                  setStep(2);
                  setSlides([]);
                  setPptFilename("");
                  setExplanations({});
                  setProcessingComplete(false);
                  setInputMethod(null);
                  setFile(null);
                  setTopic("");
                }}
              />
            )}
          </motion.div>

          <div className="dashboard-footer">
            <div className="user-profile">
              {selectedAvatar !== null && (
                <img
                  src={`/assets/${selectedAvatar === 0 ? "boy" : "girl"}.jpg`}
                  alt="Selected Avatar"
                  className="footer-avatar"
                />
              )}
              <span className="user-name">Your Account</span>
              <button className="logout-button" onClick={() => setCurrentView("landing")}>
                Logout
              </button>
            </div>
          </div>
        </div>
      )
    } else {
      return <LandingPage setCurrentView={setCurrentView} />
    }
  }

  return renderView()
}

export default Head