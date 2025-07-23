import React from "react";
import { useAtom } from "jotai";
import { userAtom } from "../store/AuthStore";
import { AuthApi } from "../api/api";

const Chat = () => {
  const [user] = useAtom(userAtom);

  const handleLogout = () => {
    AuthApi.logout();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="bg-white/10 backdrop-blur-sm border-b border-white/20 p-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
              <span className="text-white font-bold">A</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Anya</h1>
              <p className="text-sm text-gray-300">
                {user?.preferences?.news_personality?.replace("_", " ") ||
                  "AI News Assistant"}
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <span className="text-white">Hello, {user?.profile?.name}!</span>
            <button
              onClick={handleLogout}
              className="bg-white/10 hover:bg-white/20 text-white px-4 py-2 rounded-lg transition-colors"
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      {/* Chat Interface */}
      <div className="max-w-4xl mx-auto p-6">
        <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-8 border border-white/20 text-center">
          <h2 className="text-2xl font-bold text-white mb-4">
            Welcome to Anya Chat! 🎉
          </h2>
          <p className="text-gray-300 mb-6">
            Your personalized news experience is ready. The chat interface will
            be built next!
          </p>
          <div className="text-left max-w-md mx-auto bg-white/5 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-2">
              Your Preferences:
            </h3>
            <p className="text-gray-300 mb-1">
              <strong>Personality:</strong>{" "}
              {user?.preferences?.news_personality || "Not set"}
            </p>
            <p className="text-gray-300">
              <strong>Topics:</strong>{" "}
              {user?.preferences?.favorite_topics?.join(", ") ||
                "None selected"}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;
