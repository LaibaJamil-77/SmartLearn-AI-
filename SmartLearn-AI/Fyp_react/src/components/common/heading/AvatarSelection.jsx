import { motion } from "framer-motion"
import { FaArrowRight } from "react-icons/fa"

const AvatarSelection = ({ selectedAvatar, handleAvatarSelect, handleContinue }) => {
  const avatars = ["/assets/boy.jpg", "/assets/girl.jpg"]

  const handleImageError = (e) => {
    console.error("Failed to load image:", e.target.src)
    e.target.src = "https://via.placeholder.com/150?text=Avatar+Error"
  }

  return (
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
            <div className="avatar-image-wrapper">
              <img src={avatar} alt={`Avatar ${index + 1}`} onError={handleImageError} />
            </div>
            <p className="avatar-label">Avatar {index + 1}</p>
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
  )
}

export default AvatarSelection 