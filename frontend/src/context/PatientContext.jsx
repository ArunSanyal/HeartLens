import { createContext, useContext, useState, useCallback, useMemo, useEffect } from 'react';
import { fetchPatients, fetchNarrative } from '../api';

const PatientContext = createContext(null);

export function PatientProvider({ children }) {
  const [allPatients, setAllPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedPatientIds, setSelectedPatientIds] = useState(new Set());
  const [activePatient, setActivePatient] = useState(null);
  const [activeFeature, setActiveFeature] = useState(null);
  const [colorMode, setColorMode] = useState('risk');
  const [narrative, setNarrative] = useState(null);
  const [narrativeLoading, setNarrativeLoading] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    { role: 'assistant', text: 'Welcome to HeartLens! Select a patient or ask me anything about the data.' }
  ]);

  // Load patients from API on mount
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchPatients()
      .then(data => {
        if (!cancelled) {
          setAllPatients(data);
          setLoading(false);
        }
      })
      .catch(err => {
        if (!cancelled) {
          console.error('Failed to load patients:', err);
          setError(err.message);
          setLoading(false);
        }
      });
    return () => { cancelled = true; };
  }, []);

  // Fetch narrative from API when active patient changes
  useEffect(() => {
    if (!activePatient) {
      setNarrative(null);
      return;
    }
    let cancelled = false;
    setNarrativeLoading(true);
    fetchNarrative(activePatient.id)
      .then(res => {
        if (!cancelled) {
          setNarrative(res.narrative);
          setNarrativeLoading(false);
        }
      })
      .catch(err => {
        if (!cancelled) {
          console.error('Narrative fetch failed:', err);
          setNarrative('Narrative unavailable. Please check backend connection.');
          setNarrativeLoading(false);
        }
      });
    return () => { cancelled = true; };
  }, [activePatient]);

  const selectedPatients = useMemo(() => {
    if (selectedPatientIds.size === 0) return allPatients;
    return allPatients.filter(p => selectedPatientIds.has(p.id));
  }, [selectedPatientIds, allPatients]);

  const brushPatients = useCallback((ids) => {
    setSelectedPatientIds(new Set(ids));
  }, []);

  const selectPatient = useCallback((patient) => {
    setActivePatient(patient);
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedPatientIds(new Set());
    setActivePatient(null);
  }, []);

  const toggleColorMode = useCallback(() => {
    setColorMode(prev => prev === 'risk' ? 'site' : 'risk');
  }, []);

  const addChatMessage = useCallback((msg) => {
    setChatMessages(prev => [...prev, msg]);
  }, []);

  const value = {
    allPatients,
    loading,
    error,
    selectedPatients,
    selectedPatientIds,
    activePatient,
    activeFeature,
    colorMode,
    chatMessages,
    narrative,
    narrativeLoading,
    brushPatients,
    selectPatient,
    clearSelection,
    setActiveFeature,
    toggleColorMode,
    addChatMessage,
    setChatMessages,
  };

  return (
    <PatientContext.Provider value={value}>
      {children}
    </PatientContext.Provider>
  );
}

export function usePatients() {
  const ctx = useContext(PatientContext);
  if (!ctx) throw new Error('usePatients must be used within PatientProvider');
  return ctx;
}
