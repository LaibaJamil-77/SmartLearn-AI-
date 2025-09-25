import { motion } from "framer-motion"
import { FaCloudUploadAlt, FaKeyboard, FaArrowRight, FaBars, FaTimes } from "react-icons/fa"
import {
  FaComments,
  FaHistory,
  FaBookmark,
  FaUser,
  FaChalkboardTeacher,
  FaLightbulb,
  FaGraduationCap,
  FaRobot,
  FaCog,
  FaQuestionCircle,
} from "react-icons/fa"
import { useState } from "react"
import { useNavigate } from "react-router-dom"

const InputSelection = ({
  sidebarOpen,
  toggleSidebar,
  inputMethod,
  setInputMethod,
  file,
  setFile,
  topic,
  setTopic,
  handleContinue
}) => {
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  return (
    <div className="input-selection-container">
      <button className="sidebar-toggle" onClick={toggleSidebar}>
        {sidebarOpen ? <FaTimes /> : <FaBars />}
      </button>
      <motion.div
        className={`sidebar ${sidebarOpen ? "expanded" : ""}`}
        initial={{ width: 60, opacity: 1 }}
        animate={{
          width: sidebarOpen ? 250 : 60,
          opacity: 1,
        }}
        transition={{ duration: 0.3 }}
      >
        <div className="sidebar-content">
          <div className="sidebar-item">
            <FaComments />
            {sidebarOpen && <span>New Chat</span>}
          </div>
          <div className="sidebar-item">
            <FaHistory />
            {sidebarOpen && <span>History</span>}
          </div>
          <div className="sidebar-item">
            <FaBookmark />
            {sidebarOpen && <span>Saved Chats</span>}
          </div>
          <div className="sidebar-item">
            <FaUser />
            {sidebarOpen && <span>My Profile</span>}
          </div>
          <div className="sidebar-item">
            <FaChalkboardTeacher />
            {sidebarOpen && <span>My Courses</span>}
          </div>
          <div className="sidebar-item">
            <FaLightbulb />
            {sidebarOpen && <span>Learning Tips</span>}
          </div>
          <div className="sidebar-item">
            <FaGraduationCap />
            {sidebarOpen && <span>Progress Tracker</span>}
          </div>
          <div className="sidebar-item">
            <FaRobot />
            {sidebarOpen && <span>AI Tutor Settings</span>}
          </div>
          <div className="sidebar-item">
            <FaCog />
            {sidebarOpen && <span>Settings</span>}
          </div>
          <div className="sidebar-item">
            <FaQuestionCircle />
            {sidebarOpen && <span>Help</span>}
          </div>
        </div>
      </motion.div>

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
            <p>Upload a PDF, DOCX, or PPTX file</p>

            {inputMethod === "upload" && (
              <motion.div
                className="upload-area"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                transition={{ duration: 0.3 }}
              >
                <label className="file-input-label">
                  {file ? file.name : "Choose a file or drag it here"}
                  <input
                    type="file"
                    accept=".pdf,.docx,.pptx"
                    onChange={handleFileChange}
                  />
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

        <motion.button
          className={`continue-button ${
            (inputMethod === "upload" && !file) || (inputMethod === "text" && !topic) || !inputMethod
              ? "disabled"
              : ""
          }`}
          whileHover={
            (inputMethod === "upload" && file) || (inputMethod === "text" && topic) ? { scale: 1.05 } : {}
          }
          whileTap={
            (inputMethod === "upload" && file) || (inputMethod === "text" && topic) ? { scale: 0.95 } : {}
          }
          onClick={handleContinue}
          disabled={(inputMethod === "upload" && !file) || (inputMethod === "text" && !topic) || !inputMethod}
        >
          Start Learning <FaArrowRight />
        </motion.button>
      </div>
    </div>
  )
}

export default InputSelection 