import React, { useEffect } from "react";
import { GoogleLogin } from "@react-oauth/google";
import { useAtom } from "jotai";
import { jwtDecode } from "jwt-decode";
import {
  Sparkles,
  Brain,
  Mic,
  Newspaper,
  ArrowRight,
  Shield,
  ChevronRight
} from "lucide-react";
import { apiBaseUrlAtom } from "../store/ApiStore";
import {
  userAtom,
  isLoadingAtom,
  currentPersonalityAtom,
  showLoginButtonAtom
} from "../store/AuthStore";
import { AuthApi, handleApiError } from "../api/api";

const Home = () => {
  const [user, setUser] = useAtom(userAtom);
  const [isLoading, setIsLoading] = useAtom(isLoadingAtom);
  const [currentPersonality, setCurrentPersonality] = useAtom(
    currentPersonalityAtom
  );
  const [showLoginButton, setShowLoginButton] = useAtom(showLoginButtonAtom);
  const [apiBaseUrl] = useAtom(apiBaseUrlAtom);

  // Infiya's personality introductions
  const personalities = [
    {
      id: "calm_anchor",
      emoji: "🎙️",
      name: "The Calm Anchor",
      intro: "I deliver news with calm authority and clear, professional tone.",
      color: "from-blue-500 to-blue-400"
    },
    {
      id: "friendly_explainer",
      emoji: "😊",
      name: "The Friendly Explainer",
      intro:
        "I explain news like a helpful friend — clear, warm, and relatable.",
      color: "from-yellow-400 to-amber-300"
    },
    {
      id: "investigative_reporter",
      emoji: "🕵️",
      name: "The Investigative Reporter",
      intro: "I dig deep into facts, connecting dots with sharp analysis.",
      color: "from-red-500 to-rose-500"
    },
    {
      id: "youthful_trendspotter",
      emoji: "✨",
      name: "The Youthful Trendspotter",
      intro: "I cover news with a Gen-Z flair — witty, bold, and emoji-packed.",
      color: "from-pink-500 to-fuchsia-500"
    },
    {
      id: "global_correspondent",
      emoji: "🌍",
      name: "The Global Correspondent",
      intro:
        "I bring you stories from across the world with rich context and nuance.",
      color: "from-green-500 to-emerald-400"
    },
    {
      id: "ai_analyst",
      emoji: "📊",
      name: "The AI Analyst",
      intro:
        "I break down the news logically — data-driven, objective, and concise.",
      color: "from-purple-600 to-indigo-500"
    }
  ];

  const features = [
    {
      icon: Newspaper,
      title: "Real-time News",
      description: "Get the latest updates from trusted sources worldwide."
    },
    {
      icon: Sparkles,
      title: "Personalized",
      description: "Tailored content that matches your interests and style."
    }
  ];

  // Personality cycling effect
  useEffect(() => {
    const personalityInterval = setInterval(() => {
      setCurrentPersonality((prev) => (prev + 1) % personalities.length);
    }, 4000);

    const showLoginTimer = setTimeout(() => {
      setShowLoginButton(true);
    }, 2000);

    return () => {
      clearInterval(personalityInterval);
      clearTimeout(showLoginTimer);
    };
  }, [setCurrentPersonality, setShowLoginButton]);

  // Handle successful Google login
  const handleLoginSuccess = async (credentialResponse) => {
    setIsLoading(true);

    try {
      const decoded = jwtDecode(credentialResponse.credential);

      const result = await AuthApi.googleAuth({
        token: credentialResponse.credential,
        userInfo: {
          id: decoded.sub,
          email: decoded.email,
          name: decoded.name,
          picture: decoded.picture
        }
      });

      setUser(result.user);

      setUser(result.user);

      if (result.isNewUser || !result.user.onboarding_completed) {
        window.location.href = "/preferences";
      } else {
        window.location.href = "/chat";
      }
    } catch (error) {
      const errorMessage = handleApiError(error);
      alert(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoginError = (error) => {
    console.error("Google login error:", error);
    alert("Unable to sign in with Google. Please try again.");
  };

  const currentPersonalityData = personalities[currentPersonality];

  return (
    <div className="min-h-screen bg-slate-900 text-white overflow-x-hidden">
      <div className="absolute inset-0 overflow-hidden z-0">
        <div className="absolute top-0 -left-4 w-72 h-72 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-40 animate-blob"></div>
        <div className="absolute top-0 -right-4 w-72 h-72 bg-yellow-500 rounded-full mix-blend-multiply filter blur-xl opacity-40 animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-8 left-20 w-72 h-72 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-40 animate-blob animation-delay-4000"></div>
        <div className="absolute bottom-20 right-20 w-72 h-72 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-40 animate-blob animation-delay-2000"></div>
      </div>

      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 py-8">
        <div className="w-full max-w-4xl mx-auto">
          <div className="grid md:grid-cols-2 gap-10 items-center">
            {/* Left Column: Greeting and Personality */}
            <div className="text-center md:text-left">
              <div className="inline-block px-4 py-1 bg-white/10 backdrop-blur-sm rounded-full border border-white/20 mb-4">
                <p className="text-sm font-medium text-gray-300">
                  Your AI News Journalist
                </p>
              </div>
              <h1 className="text-4xl md:text-6xl font-bold mb-3">
                Hello there!
              </h1>
              <h2 className="text-3xl md:text-4xl font-bold mb-6">
                I'm{" "}
                <span
                  className={`bg-gradient-to-r ${currentPersonalityData.color} bg-clip-text text-transparent`}
                >
                  Infiya
                </span>
              </h2>
              <h3 className="text-lg font-medium text-gray-300">
                I can be your
              </h3>{" "}
              <div className="relative mb-8 bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20 shadow-lg hover:shadow-purple-800/20 transition-all duration-300">
                <div className="flex items-center space-x-4">
                  <div className="text-4xl transform hover:scale-110 transition-transform duration-300">
                    {currentPersonalityData.emoji}
                  </div>
                  <div>
                    <h3
                      className={`text-xl font-bold bg-gradient-to-r ${currentPersonalityData.color} bg-clip-text text-transparent mb-1`}
                    >
                      {currentPersonalityData.name}
                    </h3>
                    <p className="text-gray-300 text-sm italic">
                      "{currentPersonalityData.intro}"
                    </p>
                  </div>
                </div>
                <div className="flex justify-center space-x-2 mt-4">
                  {personalities.map((_, index) => (
                    <div
                      key={index}
                      className={`w-2 h-2 rounded-full transition-all duration-300 ${
                        index === currentPersonality
                          ? `bg-gradient-to-r ${personalities[index].color}`
                          : "bg-white/30"
                      }`}
                    />
                  ))}
                </div>
              </div>
            </div>

            <div className="flex flex-col space-y-8">
              <div className="space-y-4">
                {features.map((feature, index) => {
                  const Icon = feature.icon;
                  return (
                    <div
                      key={index}
                      className="flex items-start p-4 bg-white/5 backdrop-blur-sm rounded-lg border border-white/10 hover:bg-white/10 transition-all duration-300 group"
                    >
                      <div className="bg-gradient-to-br from-purple-500 to-pink-500 w-10 h-10 rounded-lg flex items-center justify-center mr-4 flex-shrink-0 group-hover:scale-110 transition-transform duration-300">
                        <Icon className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-white">
                          {feature.title}
                        </h3>
                        <p className="text-gray-300 text-sm">
                          {feature.description}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Login Section */}
              <div
                className={`transition-all duration-1000 ${
                  showLoginButton
                    ? "opacity-100 translate-y-0"
                    : "opacity-0 translate-y-8"
                }`}
              >
                <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20 shadow-lg text-center">
                  <h3 className="text-xl font-bold text-white mb-2">
                    Ready to get started?
                  </h3>
                  <p className="text-gray-300 text-sm mb-4">
                    Sign in to personalize your news experience
                  </p>

                  {isLoading ? (
                    <div className="flex items-center justify-center py-2">
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-purple-400"></div>
                      <span className="ml-2 text-white text-sm">
                        Signing you in...
                      </span>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center space-y-3">
                      <GoogleLogin
                        onSuccess={handleLoginSuccess}
                        onError={handleLoginError}
                        theme="filled_blue"
                        size="large"
                        text="continue_with"
                        shape="rectangular"
                        width="300"
                      />
                      <div className="flex items-center space-x-1 text-xs text-gray-400">
                        <Shield className="w-3 h-3" />
                        <span>Secure & Private</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Sample Questions */}
          <div className="mt-12 text-center">
            <p className="text-gray-400 text-sm mb-3">
              Try asking Infiya about:
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {[
                "Current Affairs",
                "Tech Industry",
                "Stock market",
                "Climate Change",
                "Space exploration"
              ].map((topic, index) => (
                <span
                  key={index}
                  className="bg-white/10 backdrop-blur-sm text-gray-300 px-3 py-1 rounded-full text-xs border border-white/20 hover:bg-white/20 transition-colors cursor-default flex items-center gap-1 group"
                >
                  {topic}
                  <ChevronRight className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
