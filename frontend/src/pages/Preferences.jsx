import React, { useState, useEffect } from "react";
import { useAtom } from "jotai";
import {
  Brain,
  Mic,
  User,
  Sparkles,
  Bot,
  CheckCircle,
  Heart,
  Loader2
} from "lucide-react";
import { userAtom } from "../store/AuthStore";
import { AuthApi } from "../api/api";
import { useNavigate } from "react-router-dom";

const Preferences = () => {
  const [user, setUser] = useAtom(userAtom);
  const [selectedPersonality, setSelectedPersonality] = useState(null);
  const [favoriteTopics, setFavoriteTopics] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeStep, setActiveStep] = useState(1);
  const [animateTopics, setAnimateTopics] = useState(false);
  const navigate = useNavigate();

  // Trigger topic animation after personality selection
  useEffect(() => {
    if (selectedPersonality) {
      setTimeout(() => {
        setActiveStep(2);
        setAnimateTopics(true);
      }, 300);
    }
  }, [selectedPersonality]);

  const personalities = [
    {
      id: "calm-anchor",
      name: "The Calm Anchor",
      icon: "Mic",
      description:
        "Delivers news with clarity, neutrality, and a professional tone. Perfect for serious updates and breaking headlines.",
      color: "border-blue-400"
    },
    {
      id: "friendly-explainer",
      name: "The Friendly Explainer",
      icon: "User",
      description:
        "Explains complex news like a friend would — warm, curious, and easy to understand.",
      color: "border-yellow-400"
    },
    {
      id: "investigative-reporter",
      name: "The Investigative Reporter",
      icon: "Search",
      description:
        "Digs deep into the story. Sharp, skeptical, and focused on uncovering hidden layers.",
      color: "border-red-500"
    },
    {
      id: "youthful-trendspotter",
      name: "The Youthful Trendspotter",
      icon: "Sparkles",
      description:
        "Trendy and playful. Covers stories with a Gen-Z vibe using humor, slang, and emojis.",
      color: "border-pink-400"
    },
    {
      id: "global-correspondent",
      name: "The Global Correspondent",
      icon: "Globe",
      description:
        "Brings a global lens to reporting. Informed, respectful, and context-rich storytelling.",
      color: "border-green-500"
    },
    {
      id: "ai-analyst",
      name: "The AI Analyst",
      icon: "BarChart2",
      description:
        "Analyzes the news logically using data, patterns, and critical reasoning. Minimal fluff.",
      color: "border-purple-500"
    }
  ];

  const topicSuggestions = [
    "Technology",
    "AI & Machine Learning",
    "Business",
    "Finance",
    "Politics",
    "Science",
    "Health",
    "Climate",
    "Space",
    "Startups",
    "Crypto",
    "Sports"
  ];

  const handleSavePreferences = async () => {
    if (!selectedPersonality) {
      alert("Please select a news anchor personality!");
      return;
    }

    setIsLoading(true);
    try {
      const updatedUser = await AuthApi.updateUserPreferences({
        news_personality: selectedPersonality,
        favorite_topics: favoriteTopics,
        onboarding_completed: true
      });

      setUser(updatedUser);
      navigate("/chat");
    } catch (error) {
      console.error("Failed to save preferences:", error);
      alert("Failed to save preferences. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const toggleTopic = (topic) => {
    setFavoriteTopics((prev) =>
      prev.includes(topic) ? prev.filter((t) => t !== topic) : [...prev, topic]
    );
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white p-2 sm:p-6 lg:p-8 relative">
      <div className="absolute inset-0 overflow-hidden z-0">
        <div className="absolute top-0 -left-4 w-72 h-72 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
        <div className="absolute top-0 -right-4 w-72 h-72 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob delay-700"></div>
        <div className="absolute -bottom-8 left-20 w-72 h-72 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob delay-1000"></div>
      </div>

      <div className="max-w-screen-2xl mx-auto relative z-10">
        <div className="text-center mb-10">
          <div className="inline-block px-4 py-1 bg-white/10 backdrop-blur-sm rounded-full border border-white/20 mb-4 animate-fade-in-down">
            <p className="text-sm font-medium text-gray-300">
              Personalize Your Experience
            </p>
          </div>
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold text-white mb-1 animate-fade-in">
            <span className="inline-block animate-wave origin-[70%_70%] mr-2">
              👋
            </span>
            Hi, {user?.profile.name || "there"}!
          </h1>
          <p className="text-lg ml-10 mt-1 text-gray-300 animate-fade-in-up">
            Help Infiya get to know you better
          </p>
        </div>

        <div className="space-y-8">
          {/* Step 1: Personality Selection */}
          <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 sm:p-8 border border-white/10 shadow-lg transition-all duration-500 hover:shadow-purple-500/20">
            <div className="flex items-center mb-6">
              <div
                className={`rounded-full w-8 h-8 flex items-center justify-center text-white font-bold mr-4 transition-all duration-300 ${
                  activeStep === 1 ? "bg-purple-500 scale-110" : "bg-gray-500"
                }`}
              >
                1
              </div>
              <h2 className="text-2xl font-bold text-white">
                What kind of personality would you like Infiya to have?
              </h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {personalities.map((p) => {
                const Icon = p.icon;
                const isSelected = selectedPersonality === p.id;
                return (
                  <div
                    key={p.id}
                    className={`relative p-5 rounded-xl cursor-pointer transition-all duration-300 border-2 bg-slate-800/50 hover:bg-slate-700/50 hover:scale-105 group ${
                      isSelected
                        ? `${p.color} scale-105 shadow-xl`
                        : "border-transparent"
                    }`}
                    onClick={() => setSelectedPersonality(p.id)}
                  >
                    {isSelected && (
                      <CheckCircle className="absolute top-2 right-2 w-5 h-5 text-purple-400 animate-pulse" />
                    )}
                    <div className="flex items-center space-x-4">
                      <div>
                        <h3 className="font-bold text-white group-hover:text-purple-300 transition-colors duration-300">
                          {p.name}
                        </h3>
                        <p className="text-sm text-gray-400">{p.description}</p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Step 2: Topic Selection */}
          <div
            className={`bg-white/5 backdrop-blur-sm rounded-2xl p-6 sm:p-8 border border-white/10 shadow-lg transition-all duration-500 ${
              animateTopics
                ? "opacity-100 transform translate-y-0"
                : "opacity-70 transform translate-y-4"
            } hover:shadow-purple-500/20`}
          >
            <div className="flex items-center mb-6">
              <div
                className={`rounded-full w-8 h-8 flex items-center justify-center text-white font-bold mr-4 transition-all duration-300 ${
                  activeStep === 2 ? "bg-purple-500 scale-110" : "bg-gray-500"
                }`}
              >
                2
              </div>
              <h2 className="text-2xl font-bold text-white">
                What topics are you interested in?
                <span className="text-base text-gray-400 ml-2">(Optional)</span>
              </h2>
            </div>

            <div className="flex flex-wrap gap-3">
              {topicSuggestions.map((topic, index) => {
                const isSelected = favoriteTopics.includes(topic);
                return (
                  <button
                    key={topic}
                    className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 border ${
                      isSelected
                        ? "bg-purple-500 border-purple-400 text-white"
                        : "bg-white/10 border-white/20 text-gray-300 hover:bg-white/20"
                    } ${
                      animateTopics ? "animate-fade-in-delay" : ""
                    } transition-all duration-300`}
                    style={{ animationDelay: `${index * 100}ms` }}
                    onClick={() => toggleTopic(topic)}
                  >
                    {topic}
                    {isSelected && (
                      <Heart className="inline-block ml-1 w-3 h-3" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Save Button */}
        <div className="mt-10 text-center">
          <button
            onClick={handleSavePreferences}
            disabled={isLoading || !selectedPersonality}
            className={`w-full max-w-xs px-8 py-3 rounded-xl text-lg font-semibold transition-all duration-300 transform hover:scale-105 ${
              !selectedPersonality
                ? "bg-gray-600 text-gray-400 cursor-not-allowed"
                : "bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg hover:shadow-purple-500/50"
            } ${
              selectedPersonality ? "animate-pulse-subtle" : ""
            } transition-all duration-300`}
          >
            {isLoading ? (
              <span className="flex items-center justify-center">
                <Loader2 className="animate-spin mr-2 h-5 w-5" />
                Saving...
              </span>
            ) : (
              "Start Chatting with Infiya"
            )}
          </button>
          {!selectedPersonality && (
            <p className="text-sm text-gray-400 mt-3">
              Please select a personality to continue.
            </p>
          )}
          {selectedPersonality && favoriteTopics.length > 0 && (
            <div className="mt-3 animate-fade-in">
              <p className="text-sm text-purple-300 inline-flex items-center">
                Great choices! <span className="mx-1">✨</span> Infiya can't
                wait to chat with you.
              </p>
              <div className="mt-2 flex justify-center space-x-1">
                {favoriteTopics.slice(0, 5).map((topic, i) => (
                  <span
                    key={i}
                    className="inline-block px-2 py-1 bg-purple-500/20 rounded-md text-xs text-purple-300 animate-fade-in-delay"
                    style={{ animationDelay: `${i * 150}ms` }}
                  >
                    {topic}
                  </span>
                ))}
                {favoriteTopics.length > 5 && (
                  <span
                    className="inline-block px-2 py-1 bg-purple-500/20 rounded-md text-xs text-purple-300 animate-fade-in-delay"
                    style={{ animationDelay: "750ms" }}
                  >
                    +{favoriteTopics.length - 5} more
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Preferences;
