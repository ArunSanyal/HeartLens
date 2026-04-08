import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { PatientProvider } from './context/PatientContext';
import Navbar from './components/common/Navbar';
import Footer from './components/common/Footer';
import Home from './pages/Home';
import About from './pages/About';
import Contact from './pages/Contact';
import Dashboard from './components/Layout/Dashboard';
import './App.css';

function DashboardPage() {
  return (
    <PatientProvider>
      <Dashboard />
    </PatientProvider>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/dashboard" element={<DashboardPage />} />
      </Routes>
      <Footer />
    </BrowserRouter>
  );
}

export default App;
