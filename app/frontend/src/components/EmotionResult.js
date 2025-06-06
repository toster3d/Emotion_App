import React from 'react';
import { Card, Alert } from 'react-bootstrap';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { 
  FaSmile, FaSadTear, FaAngry, FaMeh, FaFlushed, FaSurprise
} from 'react-icons/fa';

const EMOTION_COLORS = {
  anger: '#dc3545',  // red
  fear: '#6f42c1',   // purple
  happiness: '#ffc107',  // yellow
  neutral: '#0d6efd', // blue
  sadness: '#6c757d',    // gray
  surprised: '#198754', // green
};

const EMOTION_ICONS = {
  anger: <FaAngry className="emotion-icon emotion-angry" />,
  fear: <FaFlushed className="emotion-icon emotion-fear" />,
  happiness: <FaSmile className="emotion-icon emotion-happy" />,
  neutral: <FaMeh className="emotion-icon emotion-neutral" />,
  sadness: <FaSadTear className="emotion-icon emotion-sad" />,
  surprised: <FaSurprise className="emotion-icon emotion-surprised" />,
};

const EmotionResult = ({ result, error }) => {
  if (error) {
    return (
      <Alert variant="danger">
        <Alert.Heading>Error</Alert.Heading>
        <p>{error}</p>
      </Alert>
    );
  }
  
  if (!result) {
    return null;
  }
  
  // Prepare data for chart
  const chartData = Object.entries(result.probabilities || {}).map(([emotion, probability]) => ({
    name: emotion,
    value: probability * 100, // Convert to percentage
  }));
  
  // Sort by value to have more visually appealing chart
  chartData.sort((a, b) => b.value - a.value);
  
  // Custom label renderer - only show labels for emotions > 5%
  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, index, name, value }) => {
    if (value < 5) return null; // Hide labels for small segments
    
    const RADIAN = Math.PI / 180;
    const radius = outerRadius * 1.1;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);
    
    return (
      <text 
        x={x} 
        y={y} 
        fill={EMOTION_COLORS[name] || '#000'} 
        textAnchor={x > cx ? 'start' : 'end'} 
        dominantBaseline="central"
        fontSize="12"
        fontWeight="bold"
      >
        {`${name}: ${value.toFixed(1)}%`}
      </text>
    );
  };
  
  return (
    <Card>
      <Card.Body>
        <Card.Title>Emotion Analysis Results</Card.Title>
        
        <div className="text-center mb-4">
          {EMOTION_ICONS[result.emotion] || EMOTION_ICONS.neutral}
          <h3>{result.emotion}</h3>
          <p className="lead">Confidence: {(result.confidence * 100).toFixed(1)}%</p>
        </div>
        
        <div style={{ width: '100%', height: 300 }}>
          <ResponsiveContainer>
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                label={renderCustomizedLabel}
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={EMOTION_COLORS[entry.name] || '#000000'} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </Card.Body>
    </Card>
  );
};

export default EmotionResult; 