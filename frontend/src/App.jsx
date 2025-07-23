import React from "react";
import { GoogleOAuthProvider } from "@react-oauth/google";
import Home from "./pages/Home";

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;

function App() {
  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      <div className="App">
        <Home />
      </div>
    </GoogleOAuthProvider>
  );
}

export default App;
