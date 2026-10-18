import React from 'react';

export default function RiskGauge({ percentage = 0, riskLevel = "Low" }) {
  // Clamped percentage 0-100
  const pct = Math.max(0, Math.min(100, percentage));
  
  // Calculate arc parameters
  const radius = 80;
  const stroke = 14;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  // Semi-circle / 260 degree arc
  const arcLength = circumference * 0.75;
  const strokeDashoffset = arcLength - (pct / 100) * arcLength;

  const getColor = () => {
    if (pct < 20) return { stroke: "#10b981", glow: "rgba(16, 185, 129, 0.4)", text: "text-emerald-400" };
    if (pct < 45) return { stroke: "#f59e0b", glow: "rgba(245, 158, 11, 0.4)", text: "text-amber-400" };
    if (pct < 65) return { stroke: "#f97316", glow: "rgba(249, 115, 22, 0.4)", text: "text-orange-400" };
    return { stroke: "#f43f5e", glow: "rgba(244, 63, 94, 0.5)", text: "text-rose-400" };
  };

  const theme = getColor();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ position: 'relative', width: 200, height: 160, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <svg height="190" width="200" viewBox="0 0 200 190" style={{ transform: 'rotate(-135deg)', overflow: 'visible' }}>
          <defs>
            <filter id="gauge-glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="6" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="40%" stopColor="#f59e0b" />
              <stop offset="70%" stopColor="#f97316" />
              <stop offset="100%" stopColor="#f43f5e" />
            </linearGradient>
          </defs>

          {/* Background track */}
          <circle
            stroke="rgba(255, 255, 255, 0.08)"
            fill="transparent"
            strokeWidth={stroke}
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeLinecap="round"
            r={normalizedRadius}
            cx="100"
            cy="100"
          />

          {/* Dynamic value arc */}
          <circle
            stroke={theme.stroke}
            fill="transparent"
            strokeWidth={stroke}
            strokeDasharray={`${arcLength} ${circumference}`}
            style={{ 
              strokeDashoffset, 
              transition: 'stroke-dashoffset 0.8s cubic-bezier(0.16, 1, 0.3, 1), stroke 0.4s ease',
              filter: `drop-shadow(0 0 8px ${theme.glow})`
            }}
            strokeLinecap="round"
            r={normalizedRadius}
            cx="100"
            cy="100"
          />
        </svg>

        {/* Center Readout */}
        <div style={{
          position: 'absolute',
          top: '46%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center',
          pointerEvents: 'none'
        }}>
          <div style={{ 
            fontSize: '2.5rem', 
            fontWeight: '800', 
            letterSpacing: '-0.03em', 
            color: '#ffffff',
            lineHeight: 1,
            textShadow: `0 0 20px ${theme.glow}`
          }}>
            {pct.toFixed(1)}%
          </div>
          <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#94a3b8', marginTop: 4 }}>
            Readmission Risk
          </div>
        </div>
      </div>

      {/* Tier Badge */}
      <div className={`badge badge-${riskLevel.toLowerCase()}`} style={{ marginTop: -15, fontSize: '0.85rem', padding: '6px 14px' }}>
        <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: theme.stroke, display: 'inline-block' }}></span>
        {riskLevel} Risk Tier
      </div>
    </div>
  );
}
