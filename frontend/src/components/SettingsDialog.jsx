import React, { useState, useEffect } from "react";
import { X, Sparkles, Palette, Globe } from "lucide-react";
import { useAtom } from "jotai";
import { userAtom } from "../store/AuthStore";
import { AuthApi } from "../api/api";
import {
  Brain,
  Mic,
  User,
  Bot,
  CheckCircle,
  Heart,
  Loader2
} from "lucide-react";

const SettingsDialog = ({ isOpen, onClose }) => {
  const [user, setUser] = useAtom(userAtom);
  const [preferences, setPreferences] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    console.log("the user is : ", user);
    //     {
    //     "id": "6881d9ac249fc004b6bf699a",
    //     "google_id": "112507674337596503347",
    //     "profile": {
    //         "name": "Divyansh",
    //         "email": "dvmuley10@gmail.com",
    //         "picture": "https://lh3.googleusercontent.com/a/ACg8ocJFE1n0ixL6qBEexfx8pHefUuinDDhnjk0CO1F63FEqnqszWnoA=s96-c",
    //         "bio": null,
    //         "location": null,
    //         "timezone": "UTC"
    //     },
    //     "preferences": {
    //         "news_personality": "friendly-explainer",
    //         "favorite_topics": [
    //             "finance",
    //             "ai & machine learning",
    //             "business",
    //             "technology"
    //         ],
    //         "content_length": "medium",
    //         "notification_settings": {
    //             "email": true,
    //             "browser_push": false,
    //             "daily_digest": true
    //         },
    //         "ui_theme": "light",
    //         "language": "en"
    //     },
    //     "stats": {
    //         "total_conversations": 0,
    //         "total_messages": 0,
    //         "total_time_spent": 0,
    //         "favorite_topics_usage": {},
    //         "last_active": "2025-07-24T07:02:40.204000",
    //         "streak_days": 0,
    //         "total_sources_read": 0
    //     },
    //     "is_active": true,
    //     "onboarding_completed": true,
    //     "subscription_tier": "free",
    //     "created_at": "2025-07-24T06:58:52.389000",
    //     "last_login": "2025-07-24T07:02:39.812000"
    // }

    if (isOpen && user?.preferences) {
      setPreferences({
        news_personality: user.preferences.news_personality || 'friendly-explainer',
        favorite_topics: user.preferences.favorite_topics || [],
        content_length: user.preferences.content_length || 'medium'
      });
      setHasChanges(false);
    }
  }, [isOpen, user]);

  const personalityOptions = [
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

  const topicOptions = [
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

  const sourceOptions = [
    "Reuters",
    "Associated Press",
    "BBC",
    "CNN",
    "The Guardian",
    "Wall Street Journal",
    "New York Times",
    "NPR",
    "Bloomberg"
  ];

  const handlePreferenceChange = (key, value) => {
    setPreferences((prev) => ({
      ...prev,
      [key]: value
    }));
    setHasChanges(true);
  };

  const handleTopicToggle = (topic) => {
    const currentTopics = preferences.favorite_topics || [];
    const normalizedTopic = topic.toLowerCase();
    const newTopics = currentTopics.includes(normalizedTopic)
      ? currentTopics.filter((t) => t !== normalizedTopic)
      : [...currentTopics, normalizedTopic];

    handlePreferenceChange("favorite_topics", newTopics);
  };



  const handleSave = async () => {
    setIsLoading(true);
    try {
      // Update preferences via API
      const updatedUser = await AuthApi.updateUserPreferences(preferences);

      // Update local user state
      setUser(updatedUser);

      setHasChanges(false);
      onClose();
    } catch (error) {
      console.error("Failed to update preferences:", error);
      alert("Failed to save preferences. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    if (hasChanges) {
      if (
        window.confirm(
          "You have unsaved changes. Are you sure you want to close?"
        )
      ) {
        onClose();
      }
    } else {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={handleClose}
      />

      {/* Dialog */}
      <div className="relative bg-slate-800 rounded-2xl border border-white/10 w-full max-w-2xl max-h-[90vh] overflow-hidden animate-fade-in-up">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/10">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
              <User className="w-4 h-4 text-white" />
            </div>
            <h2 className="text-xl font-bold text-white">Settings</h2>
          </div>
          <button
            onClick={handleClose}
            className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)] custom-scrollbar">
          <div className="space-y-8">
            {/* News Personality */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Sparkles className="w-5 h-5 text-purple-400" />
                <h3 className="text-lg font-semibold text-white">
                  News Personality
                </h3>
              </div>
              <div className="grid gap-3">
                {personalityOptions.map((option) => {
                  const Icon = option.icon;

                  return (
                    <label
                      key={option.id}
                      className={`flex items-start gap-3 p-4 rounded-xl border-2 cursor-pointer transition-all ${
                        preferences.news_personality === option.id
                          ? `${option.color} bg-white/5`
                          : "border-white/10 hover:border-white/20 bg-slate-700/50"
                      }`}
                    >
                      <input
                        type="radio"
                        name="personality"
                        value={option.id}
                        checked={preferences.news_personality === option.id}
                        onChange={(e) =>
                          handlePreferenceChange(
                            "news_personality",
                            e.target.value
                          )
                        }
                        className="mt-1 accent-purple-500"
                      />
                      <div>
                        <div className="flex items-center gap-2 text-white font-medium">
                          {option.name}
                        </div>
                        <div className="text-sm text-gray-400">
                          {option.description}
                        </div>
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>

            {/* Topics of Interest */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Globe className="w-5 h-5 text-blue-400" />
                <h3 className="text-lg font-semibold text-white">
                  Topics of Interest
                </h3>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {topicOptions.map((topic) => {
                  const isSelected = (preferences.favorite_topics || []).includes(topic.toLowerCase());
                  return (
                    <label
                      key={topic}
                      className={`flex items-center gap-2 p-3 rounded-lg border cursor-pointer transition-all ${
                        isSelected
                          ? "border-blue-500 bg-blue-500/10 text-blue-300"
                          : "border-white/10 hover:border-white/20 text-gray-300"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => handleTopicToggle(topic)}
                        className="text-blue-500"
                      />
                      <span className="text-sm font-medium">{topic}</span>
                    </label>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 pt-2 pb-2 pr-6 border-t border-white/10 bg-slate-800/50">
          <button
            onClick={handleClose}
            className="px-4 py-2 text-gray-300 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={!hasChanges || isLoading}
            className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105"
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Saving...
              </div>
            ) : (
              "Save Changes"
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsDialog;
