import { motion } from "framer-motion"
import { FaFacebook, FaInstagram, FaTwitter, FaYoutube, FaEnvelope } from "react-icons/fa"

const LandingPage = ({ setCurrentView }) => {
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
          <div className="about-image">
            <img src="/assets/office.jpg" alt="About Us" />
          </div>

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
          <div className="testimonial-card">
            <img src="/assets/user1.jpg" alt="samaviya" className="user-img" />
            <div className="testimonial-content">
              <h3 className="user-name">Ms Sama</h3>
              <p className="user-role">Student</p>
              <div className="stars">⭐⭐⭐⭐⭐</div>
              <p className="testimonial-text">"As a teacher, this tool saves me hours of work!"</p>
            </div>
          </div>

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

export default LandingPage 