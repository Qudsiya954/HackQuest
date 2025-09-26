import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

// TODO: Replace the following with your app's Firebase project configuration.
// You can find this in your Firebase project settings under the "General" tab.
const firebaseConfig = {
  apiKey: "AIzaSyATYTL_jXDdthprkQJfAvmyX8c6rW73FpI",
  authDomain: "hacktrix-mental-health.firebaseapp.com",
  projectId: "hacktrix-mental-health",
  storageBucket: "hacktrix-mental-health.firebasestorage.app",
  messagingSenderId: "728167801395",
  appId: "1:728167801395:web:d150ae6112a35672787946"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Export the necessary Firebase services to be used throughout your app
export const db = getFirestore(app);
export const auth = getAuth(app);

// Export the main app instance as a default export
export default app;
