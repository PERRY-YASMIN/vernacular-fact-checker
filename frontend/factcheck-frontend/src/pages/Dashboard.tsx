import React from "react";
import Header from "../components/Header";
import SearchBar from "../components/SearchBar";
import ClaimCard from "../components/ClaimCard";
import "../index.css";

// Mock claims covering milestones 2–4
const mockClaims = [
  { text: "The population of Delhi is 20 million in 2026.", verdict: "Refuted", confidence: 92, language: "English" },
  { text: "भारत में साल 2023 में चुनाव होंगे।", verdict: "Supported", confidence: 88, language: "Hindi" },
  { text: "Fuel price increased by 10% in March.", verdict: "Supported", confidence: 85, language: "English" },
  { text: "यह अफवाह है कि कोरोना का इलाज दवा X से हो जाएगा।", verdict: "Refuted", confidence: 95, language: "Hindi" },
];

const Dashboard: React.FC = () => {
  const [claims, setClaims] = React.useState(mockClaims);

  const handleSearch = (query: string) => {
    const filtered = mockClaims.filter(c => c.text.toLowerCase().includes(query.toLowerCase()));
    setClaims(filtered);
  };

  return (
    <div className="dashboard-bg">
      <Header />
      <SearchBar onSearch={handleSearch} />
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
        {claims.map((claim, idx) => (
          <ClaimCard
            key={idx}
            {...claim}
            onApprove={() => alert("Approved!")}
            onReject={() => alert("Rejected!")}
          />
        ))}
      </div>
    </div>
  );
};

export default Dashboard;
