import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { PatientProvider } from './context/PatientContext';
import Navbar from './components/common/Navbar';
import Footer from './components/common/Footer';
import Home from './pages/Home';
import About from './pages/About';
import Contact from './pages/Contact';
import Dashboard from './components/Layout/Dashboard';
import PatientDeepDive from './pages/PatientDeepDive';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <PatientProvider>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/patient/:id" element={<PatientDeepDive />} />
        </Routes>
        <Footer />
      </PatientProvider>
    </BrowserRouter>
  );
}

export default App;
