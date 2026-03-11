import React from "react";

interface ClaimCardProps {
  text: string;
  verdict?: string;
  confidence?: number;
  language?: string;
  onApprove?: () => void;
  onReject?: () => void;
}

const ClaimCard: React.FC<ClaimCardProps> = ({
  text, verdict, confidence, language, onApprove, onReject
}) => {
  return (
    <div className="claim-card">
      <p>{text}</p>
      {verdict && (
        <div className="verdict">
          <span><b>Verdict:</b> {verdict}</span>
          <span>Confidence: {confidence}%</span>
        </div>
      )}
      {language && <span className="language">Lang: {language}</span>}
      {(onApprove || onReject) && (
        <div className="buttons">
          {onApprove && <button className="approve" onClick={onApprove}>Approve</button>}
          {onReject && <button className="reject" onClick={onReject}>Reject</button>}
        </div>
      )}
    </div>
  );
};

export default ClaimCard;
