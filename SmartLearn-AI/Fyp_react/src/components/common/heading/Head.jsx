"use client"
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
  const [file, setFile] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // New state for showing slides after processing
  const [processingComplete, setProcessingComplete] = useState(false)

  // Add this near the top with your other state variables
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString())

  const avatars = ["/assets/ladiavatar.jpg", "/assets/gentavatar.jpg"]

  // Add this useEffect hook after your other state declarations
  useEffect(() => {
    // Update the clock every second
    const clockInterval = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString())
    }, 1000)

    // Clean up the interval on component unmount
    return () => clearInterval(clockInterval)
  }, [])

  // Toggle between login and signup forms
  const toggleForm = () => {
    setIsLogin(!isLogin)
  }

  // Handle login/signup submission
  const handleAuthSubmit = (e) => {
    e.preventDefault()
    // Here you would typically handle authentication
    // For now, just move to dashboard
    setCurrentView("dashboard")
  }

  const handleAvatarSelect = (index) => {
    setSelectedAvatar(index)
  }

  const handleContinue = () => {
    if (step === 1 && selectedAvatar !== null) {
      setStep(2)
    } else if (step === 2 && inputMethod) {
      if (inputMethod === "upload" && file) {
        // Process file upload
        console.log("Processing file:", file)
        // Here you would typically upload the file to your server
      } else if (inputMethod === "text" && topic) {
        // Process topic
        console.log("Processing topic:", topic)
        // Here you would typically send the topic to your AI for processing
      }
      // Move to next step or redirect to results page
      setStep(3)

      // Simulate processing time and then show slides
      setTimeout(() => {
        setProcessingComplete(true)
      }, 3000)
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen)
  }

  // Render different views based on state
  const renderView = () => {
    if (currentView === "login") {
      return (
        <div className="login-container">
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
              <div className="avatar-selection">
                <h2>Choose Your Avatar</h2>
                <p>Select an avatar that represents you in the SmartLearn community</p>

                <div className="avatars-grid">
                  {avatars.map((avatar, index) => (
                    <motion.div
                      key={index}
                      className={`avatar-item ${selectedAvatar === index ? "selected" : ""}`}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => handleAvatarSelect(index)}
                    >
                      <img src={avatar || "/placeholder.svg"} alt={`Avatar ${index + 1}`} />
                      {selectedAvatar === index && <div className="selected-indicator">✓</div>}
                    </motion.div>
                  ))}
                </div>

                <motion.button
                  className={`continue-button ${selectedAvatar === null ? "disabled" : ""}`}
                  whileHover={selectedAvatar !== null ? { scale: 1.05 } : {}}
                  whileTap={selectedAvatar !== null ? { scale: 0.95 } : {}}
                  onClick={handleContinue}
                  disabled={selectedAvatar === null}
                >
                  Continue <FaArrowRight />
                </motion.button>
              </div>
            )}

            {step === 2 && (
              <div className="input-selection-container">
                {/* Only render the main input selection UI, not the sidebar or landing page */}
                <div className="input-selection">
                  <h2>How Would You Like to Learn?</h2>
                  <p>Upload a document or enter a topic to get started</p>
                  <div className="input-methods">
                    <motion.div
                      className={`input-method ${inputMethod === "upload" ? "selected" : ""}`}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => setInputMethod("upload")}
                    >
                      <div className="method-icon">
                        <FaCloudUploadAlt />
                      </div>
                      <h3>Upload Document</h3>
                      <p>Upload a PDF, DOCX, or image file</p>

                      {inputMethod === "upload" && (
                        <motion.div
                          className="upload-area"
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          transition={{ duration: 0.3 }}
                        >
                          <label className="file-input-label">
                            {file ? file.name : "Choose a file or drag it here"}
                            <input type="file" accept=".pdf,.docx,.doc,.jpg,.jpeg,.png" onChange={handleFileChange} />
                          </label>
                        </motion.div>
                      )}
                    </motion.div>

                    <motion.div
                      className={`input-method ${inputMethod === "text" ? "selected" : ""}`}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => setInputMethod("text")}
                    >
                      <div className="method-icon">
                        <FaKeyboard />
                      </div>
                      <h3>Enter Topic</h3>
                      <p>Type a subject or question</p>

                      {inputMethod === "text" && (
                        <motion.div
                          className="topic-input-area"
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          transition={{ duration: 0.3 }}
                        >
                          <input
                            type="text"
                            placeholder="e.g., Photosynthesis, Quantum Physics"
                            value={topic}
                            onChange={(e) => setTopic(e.target.value)}
                          />
                        </motion.div>
                      )}
                    </motion.div>
                  </div>
                </div>
              </div>
            )}

            {step === 3 && (
              <div className="processing-view-container">
                {/* Only render the main processing/slides UI, not the sidebar or landing page */}
                {/* You can render your slides or processing view here if needed */}
              </div>
            )}
          </motion.div>

          <div className="dashboard-footer">
            <div className="user-profile">
              {selectedAvatar !== null && (
                <img
                  src={avatars[selectedAvatar] || "/placeholder.svg"}
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
      // Landing Page (default)
      return (
        <div className="landing-page">
          {/* Hero Section */}
          <header className="hero-section">
            <motion.div initial={{ opacity: 0, y: -50 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 1 }}>
              <h1>SmartLearn AI – Your Intelligent Virtual Tutor</h1>
              <p>Get instant answers, interactive explanations, and AI-generated slides in real time.</p>

              <div className="hero-buttons">
                <motion.button
                  className="cta-button"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setCurrentView("login")}
                >
                  Get Started
                </motion.button>
                <motion.button
                  className="login-button"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setCurrentView("login")}
                >
                  Login
                </motion.button>
              </div>
            </motion.div>
          </header>

          <section className="about-us">
            <div className="about-container">
              {/* Left Side: Image */}
              <div className="about-image">
                <img src="/assets/office.jpg" alt="About Us" />
              </div>

              {/* Right Side: Text Content */}
              <div className="about-content">
                <h2 className="about-title">ABOUT US</h2>
                <p className="about-description">
                  SmartLearn AI is a cutting-edge learning platform designed to make complex topics easier for students
                  and professionals. We provide interactive AI-driven learning experiences that enhance understanding
                  and retention.
                </p>

                <h3 className="why-choose-title">Why Choose Us?</h3>
                <ul className="why-choose-list">
                  <li>AI-powered interactive learning</li>
                  <li>Personalized tutoring in Urdu & English</li>
                  <li>Covers multiple subjects with real-time interaction</li>
                  <li>User-friendly and engaging platform</li>
                </ul>
              </div>
            </div>
          </section>

          {/* Steps Section */}
          <section className="how-it-works">
            <h2>Watch How It Works</h2>
            <motion.a
              href="https://www.example.com/tutorial-video"
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ scale: 1.1 }}
            >
              <button className="video-button">▶ Watch Tutorial</button>
            </motion.a>

            {/* Steps (Now inside the same section) */}
            <div className="steps-container">
              {[
                "Type your question or upload a file.",
                "AI analyzes and generates a response.",
                "Receive slides or a detailed explanation.",
                "Download the PPT and keep learning!",
              ].map((step, index) => (
                <motion.div
                  key={index}
                  className="step"
                  initial={{ opacity: 0, x: -50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.2 }}
                >
                  <span className="step-number">✅</span>
                  <p>{step}</p>
                </motion.div>
              ))}
            </div>
          </section>

          <section className="testimonials">
            <h2 className="section-title">
              <span className="highlight">WHAT ARE THEY SAYING</span>
            </h2>
            <div className="testimonial-container">
              {/* First Testimonial */}
              <div className="testimonial-card">
                <img src="/assets/user1.jpg" alt="samaviya" className="user-img" />
                <div className="testimonial-content">
                  <h3 className="user-name">Ms Sama</h3>
                  <p className="user-role">Student</p>
                  <div className="stars">⭐⭐⭐⭐⭐</div>
                  <p className="testimonial-text">"As a teacher, this tool saves me hours of work!"</p>
                </div>
              </div>

              {/* Second Testimonial */}
              <div className="testimonial-card">
                <img src="/assets/user2.webp" alt="Bella" className="user-img" />
                <div className="testimonial-content">
                  <h3 className="user-name">Bella</h3>
                  <p className="user-role">Student</p>
                  <div className="stars">⭐⭐⭐⭐⭐</div>
                  <p className="testimonial-text">"SmartLearn AI helped me understand complex topics easily!"</p>
                </div>
              </div>
            </div>
          </section>

          <section className="cta">
            <h2>Start Learning with SmartLearn AI Today!</h2>
            <p>Experience AI-powered tutoring with real-time explanations and slides.</p>
            <button className="cta-button" onClick={() => setCurrentView("login")}>
              Start for Free
            </button>

            <div className="faq">
              <h3>FAQs</h3>
              <p>
                <strong>Is it free?</strong> Yes! Basic features are free to use.
              </p>
              <p>
                <strong>How does AI generate slides?</strong> AI analyzes your query and auto-creates presentations.
              </p>
            </div>
          </section>

          <footer className="footer">
            <p>Follow us on:</p>
            <div className="social-icons">
              <a href="https://facebook.com" target="_blank" rel="noopener noreferrer">
                <span className="icon">
                  <FaFacebook />
                </span>{" "}
                Facebook
              </a>
              <a href="https://instagram.com" target="_blank" rel="noopener noreferrer">
                <span className="icon">
                  <FaInstagram />
                </span>{" "}
                Instagram
              </a>
              <a href="https://twitter.com" target="_blank" rel="noopener noreferrer">
                <span className="icon">
                  <FaTwitter />
                </span>{" "}
                Twitter
              </a>
              <a href="https://youtube.com" target="_blank" rel="noopener noreferrer">
                <span className="icon">
                  <FaYoutube />
                </span>{" "}
                YouTube
              </a>
            </div>
            <div className="contact-info">
              <a href="mailto:support@smartlearn.ai">
                <span className="icon">
                  <FaEnvelope />
                </span>{" "}
                Contact us: support@smartlearn.ai
              </a>
            </div>
          </footer>
        </div>
      )
    }
  }

  // return renderView()
}

export default Head

