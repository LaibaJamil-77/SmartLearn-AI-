import { useState } from "react"
import { FaGoogle, FaFacebook } from "react-icons/fa"

const LoginForm = ({ setCurrentView, toggleForm }) => {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [errorMessage, setErrorMessage] = useState("")
  const [loading, setLoading] = useState(false)

  const handleAuthSubmit = async (e) => {
    e.preventDefault()
    setErrorMessage("")
    setLoading(true)

    try {
      const response = await fetch("https://fyp-chatbot-vz6d.onrender.com/token", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({
          username: email,
          password: password,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Login failed")
      }

      const data = await response.json()
      console.log("Login success:", data)
      localStorage.setItem("token", data.access_token)
      setCurrentView("dashboard")
    } catch (error) {
      console.error("Login error:", error)
      setErrorMessage(error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-form">
      <h2>Welcome to SmartLearn AI</h2>
      <p className="auth-subtitle">Please sign in to your account and start learning</p>

      <form onSubmit={handleAuthSubmit}>
        <div className="form-group">
          <label>Email or Username</label>
          <input
            type="email"
            placeholder="Enter your email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label>Password</label>
          <input
            type="password"
            placeholder="Enter your password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>
        {errorMessage && <p className="error-message">{errorMessage}</p>}
        <div className="form-options">
          <label className="remember-me">
            <input type="checkbox" /> Remember Me
          </label>
          <button
            type="button"
            className="forgot-password"
            onClick={() => alert("Forgot Password functionality coming soon!")}
          >
            Forgot Password?
          </button>
        </div>
        <button type="submit" className="auth-button" disabled={loading}>
          {loading ? "Signing in..." : "Sign In"}
        </button>
      </form>

      <div className="divider">
        <span>or</span>
      </div>

      <div className="social-auth">
        <button className="social-button google">
          <FaGoogle /> Sign in with Google
        </button>
        <button className="social-button facebook">
          <FaFacebook /> Sign in with Facebook
        </button>
      </div>

      <p className="switch-form">
        New on our platform?
        <button type="button" onClick={toggleForm} className="switch-button">
          Create an account
        </button>
      </p>

      <button type="button" onClick={() => setCurrentView("landing")} className="back-to-home">
        Back to Home
      </button>
    </div>
  )
}

export default LoginForm 