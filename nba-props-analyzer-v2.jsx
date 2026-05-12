import React, { useState, useRef, useEffect } from 'react';

// Simulated player database with realistic stats
const PLAYER_DATABASE = {
  'tyrese haliburton': {
    id: 1630169,
    name: 'Tyrese Haliburton',
    team: 'IND',
    teamFull: 'Indiana Pacers',
    position: 'PG',
    seasonAvg: 21.8,
    last10: [24, 18, 29, 22, 19, 31, 20, 17, 25, 23],
    homeAvg: 23.2,
    awayAvg: 20.1,
    stdDev: 5.2,
    minutesAvg: 34.5,
    usage: 28.4,
  },
  'luka doncic': {
    id: 1629029,
    name: 'Luka Dončić',
    team: 'DAL',
    teamFull: 'Dallas Mavericks',
    position: 'PG',
    seasonAvg: 33.2,
    last10: [35, 28, 41, 38, 29, 32, 44, 30, 36, 27],
    homeAvg: 34.8,
    awayAvg: 31.4,
    stdDev: 5.8,
    minutesAvg: 37.2,
    usage: 36.1,
  },
  'jayson tatum': {
    id: 1628369,
    name: 'Jayson Tatum',
    team: 'BOS',
    teamFull: 'Boston Celtics',
    position: 'SF',
    seasonAvg: 27.4,
    last10: [31, 24, 28, 33, 22, 29, 26, 35, 27, 30],
    homeAvg: 28.9,
    awayAvg: 25.8,
    stdDev: 4.1,
    minutesAvg: 36.1,
    usage: 30.2,
  },
  'shai gilgeous-alexander': {
    id: 1628983,
    name: 'Shai Gilgeous-Alexander',
    team: 'OKC',
    teamFull: 'Oklahoma City Thunder',
    position: 'SG',
    seasonAvg: 31.5,
    last10: [34, 29, 38, 27, 33, 31, 35, 28, 40, 30],
    homeAvg: 32.8,
    awayAvg: 30.1,
    stdDev: 4.3,
    minutesAvg: 34.8,
    usage: 32.5,
  },
  'anthony edwards': {
    id: 1630162,
    name: 'Anthony Edwards',
    team: 'MIN',
    teamFull: 'Minnesota Timberwolves',
    position: 'SG',
    seasonAvg: 26.8,
    last10: [32, 21, 28, 35, 24, 27, 30, 19, 33, 26],
    homeAvg: 28.2,
    awayAvg: 25.3,
    stdDev: 5.5,
    minutesAvg: 35.4,
    usage: 31.8,
  },
  'corey kispert': {
    id: 1630557,
    name: 'Corey Kispert',
    team: 'WAS',
    teamFull: 'Washington Wizards',
    position: 'SF',
    seasonAvg: 10.2,
    last10: [12, 8, 14, 9, 11, 7, 13, 10, 15, 8],
    homeAvg: 11.1,
    awayAvg: 9.2,
    stdDev: 2.8,
    minutesAvg: 26.3,
    usage: 16.2,
  },
  'lebron james': {
    id: 2544,
    name: 'LeBron James',
    team: 'LAL',
    teamFull: 'Los Angeles Lakers',
    position: 'SF',
    seasonAvg: 25.1,
    last10: [28, 22, 31, 24, 27, 19, 30, 26, 23, 29],
    homeAvg: 26.4,
    awayAvg: 23.7,
    stdDev: 4.0,
    minutesAvg: 35.2,
    usage: 29.1,
  },
  'stephen curry': {
    id: 201939,
    name: 'Stephen Curry',
    team: 'GSW',
    teamFull: 'Golden State Warriors',
    position: 'PG',
    seasonAvg: 24.8,
    last10: [29, 18, 35, 22, 27, 31, 20, 26, 33, 24],
    homeAvg: 26.5,
    awayAvg: 22.9,
    stdDev: 5.9,
    minutesAvg: 32.1,
    usage: 28.9,
  }
};

// Simulated odds data (would come from The Odds API in production)
const simulateOddsData = (player) => {
  const recentAvg = player.last10.reduce((a, b) => a + b, 0) / player.last10.length;
  const projectedLine = Math.round(recentAvg * 2) / 2; // Round to nearest 0.5
  
  // Simulate different books with slight variations
  const books = [
    { key: 'draftkings', name: 'DraftKings' },
    { key: 'fanduel', name: 'FanDuel' },
    { key: 'betmgm', name: 'BetMGM' },
    { key: 'caesars', name: 'Caesars' },
    { key: 'pointsbet', name: 'PointsBet' },
  ];
  
  return books.map(book => {
    // Add some variance to simulate different book pricing
    const variance = (Math.random() - 0.5) * 10;
    const overOdds = Math.round(-110 + variance);
    const underOdds = Math.round(-110 - variance);
    
    return {
      bookmaker: book.name,
      bookmakerKey: book.key,
      line: projectedLine,
      overOdds: overOdds,
      underOdds: underOdds,
      lastUpdate: new Date().toISOString()
    };
  });
};

// Simulated game data
const simulateGameData = (player) => {
  const opponents = ['Hawks', 'Celtics', 'Lakers', 'Warriors', 'Heat', 'Suns', 'Nuggets', 'Bucks'];
  const opponent = opponents[Math.floor(Math.random() * opponents.length)];
  const isHome = Math.random() > 0.5;
  
  return {
    gameId: `game_${Date.now()}`,
    homeTeam: isHome ? player.teamFull : `${opponent}`,
    awayTeam: isHome ? `${opponent}` : player.teamFull,
    commenceTime: new Date(Date.now() + 3600000 * 4).toISOString(), // 4 hours from now
    isHome: isHome
  };
};

// Utility functions for betting math
const americanToImpliedProb = (odds) => {
  if (odds < 0) {
    return Math.abs(odds) / (Math.abs(odds) + 100);
  } else {
    return 100 / (odds + 100);
  }
};

const americanToDecimal = (odds) => {
  if (odds < 0) {
    return 1 + (100 / Math.abs(odds));
  } else {
    return 1 + (odds / 100);
  }
};

const calculateEV = (trueProbability, decimalOdds, stake = 100) => {
  const winAmount = stake * (decimalOdds - 1);
  const ev = (trueProbability * winAmount) - ((1 - trueProbability) * stake);
  return ev;
};

const calculateKellyCriterion = (trueProbability, decimalOdds) => {
  const q = 1 - trueProbability;
  const b = decimalOdds - 1;
  const kelly = ((trueProbability * b) - q) / b;
  return Math.max(0, kelly);
};

// Estimate true probability based on player stats
const estimateTrueProbability = (player, line, isOver) => {
  const { seasonAvg, last10, stdDev } = player;
  const recentAvg = last10.reduce((a, b) => a + b, 0) / last10.length;
  
  // Weight recent performance more heavily
  const projectedPoints = (seasonAvg * 0.4) + (recentAvg * 0.6);
  
  // Calculate z-score
  const zScore = (line - projectedPoints) / stdDev;
  
  // Convert z-score to probability using approximation of normal CDF
  const cdf = (z) => {
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    
    const sign = z < 0 ? -1 : 1;
    z = Math.abs(z) / Math.sqrt(2);
    
    const t = 1.0 / (1.0 + p * z);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-z * z);
    
    return 0.5 * (1.0 + sign * y);
  };
  
  const probUnder = cdf(zScore);
  return isOver ? (1 - probUnder) : probUnder;
};

const calculateHitRate = (games, line, isOver) => {
  const hits = games.filter(g => isOver ? g > line : g < line).length;
  return hits / games.length;
};

const getVarianceRating = (stdDev, avg) => {
  const cv = stdDev / avg;
  if (cv < 0.15) return { rating: 'Low', color: '#10b981', description: 'Very consistent scorer' };
  if (cv < 0.22) return { rating: 'Medium', color: '#f59e0b', description: 'Moderate game-to-game variance' };
  return { rating: 'High', color: '#ef4444', description: 'Volatile, big swings possible' };
};

const getConfidenceRating = (edge, hitRate, variance) => {
  let score = 0;
  if (edge > 5) score += 3;
  else if (edge > 2) score += 2;
  else if (edge > 0) score += 1;
  
  if (hitRate > 0.65) score += 3;
  else if (hitRate > 0.55) score += 2;
  else if (hitRate > 0.45) score += 1;
  
  if (variance.rating === 'Low') score += 2;
  else if (variance.rating === 'Medium') score += 1;
  
  if (score >= 7) return { rating: 'A', color: '#10b981', label: 'Strong Play' };
  if (score >= 5) return { rating: 'B', color: '#22c55e', label: 'Solid Play' };
  if (score >= 3) return { rating: 'C', color: '#f59e0b', label: 'Marginal' };
  return { rating: 'D', color: '#ef4444', label: 'Avoid' };
};

// Parse user input - now simpler, just need player name and prop type
const parseInput = (input) => {
  const normalized = input.toLowerCase().trim();
  
  // Match patterns like "tyrese haliburton points" or just "haliburton points"
  const patterns = [
    /^(.+?)\s+(points|pts)$/i,
    /^(.+?)$/i // fallback - just player name
  ];
  
  for (const pattern of patterns) {
    const match = normalized.match(pattern);
    if (match) {
      let playerName = match[1].trim();
      // Remove "points" or "pts" if it got included
      playerName = playerName.replace(/\s*(points|pts)$/i, '').trim();
      return { playerName, propType: 'points' };
    }
  }
  
  return null;
};

// Main analysis function
const analyzeProps = (input) => {
  const parsed = parseInput(input);
  
  if (!parsed) {
    return {
      type: 'error',
      message: "I couldn't parse that. Try format: \"Player Name points\"\n\nExample: \"Tyrese Haliburton points\""
    };
  }
  
  const { playerName } = parsed;
  const player = PLAYER_DATABASE[playerName];
  
  if (!player) {
    const availablePlayers = Object.values(PLAYER_DATABASE).map(p => p.name).join(', ');
    return {
      type: 'error',
      message: `Player "${playerName}" not found in database.\n\nAvailable players for demo: ${availablePlayers}`
    };
  }
  
  // Simulate fetching odds from API
  const oddsData = simulateOddsData(player);
  const gameData = simulateGameData(player);
  
  // Find best odds
  const bestOver = oddsData.reduce((best, curr) => 
    curr.overOdds > best.overOdds ? curr : best, oddsData[0]);
  const bestUnder = oddsData.reduce((best, curr) => 
    curr.underOdds > best.underOdds ? curr : best, oddsData[0]);
  
  const line = oddsData[0].line;
  
  // Calculate analysis for both sides
  const analyzeForSide = (isOver, bestOdds) => {
    const odds = isOver ? bestOdds.overOdds : bestOdds.underOdds;
    const impliedProb = americanToImpliedProb(odds);
    const decimalOdds = americanToDecimal(odds);
    const trueProbability = estimateTrueProbability(player, line, isOver);
    const ev = calculateEV(trueProbability, decimalOdds);
    const kellyFraction = calculateKellyCriterion(trueProbability, decimalOdds);
    const hitRate = calculateHitRate(player.last10, line, isOver);
    const recentAvg = player.last10.reduce((a, b) => a + b, 0) / player.last10.length;
    const projectedPoints = (player.seasonAvg * 0.4) + (recentAvg * 0.6);
    const variance = getVarianceRating(player.stdDev, player.seasonAvg);
    const edge = (trueProbability - impliedProb) * 100;
    const confidence = getConfidenceRating(edge, hitRate, variance);
    
    return {
      impliedProb,
      trueProbability,
      ev,
      kellyFraction,
      hitRate,
      recentAvg,
      projectedPoints,
      variance,
      edge,
      confidence,
      decimalOdds,
      bestBook: bestOdds.bookmaker,
      bestOdds: odds
    };
  };
  
  const overAnalysis = analyzeForSide(true, bestOver);
  const underAnalysis = analyzeForSide(false, bestUnder);
  
  // Determine recommendation
  const recommendation = overAnalysis.ev > underAnalysis.ev ? 'over' : 'under';
  const bestAnalysis = recommendation === 'over' ? overAnalysis : underAnalysis;
  
  return {
    type: 'analysis',
    player,
    gameData,
    line,
    oddsData,
    bestOver,
    bestUnder,
    overAnalysis,
    underAnalysis,
    recommendation,
    bestAnalysis
  };
};

// Components
const OddsTable = ({ oddsData, line, bestOver, bestUnder }) => {
  return (
    <div className="bg-gray-800/30 rounded-xl p-4 border border-gray-700/50 overflow-x-auto">
      <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
        <span className="w-2 h-2 bg-cyan-400 rounded-full"></span>
        Line Shopping - Points {line}
      </h4>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-400 border-b border-gray-700">
            <th className="text-left py-2">Book</th>
            <th className="text-center py-2">Over</th>
            <th className="text-center py-2">Under</th>
          </tr>
        </thead>
        <tbody>
          {oddsData.map((book, i) => (
            <tr key={i} className="border-b border-gray-800">
              <td className="py-2 text-white">{book.bookmaker}</td>
              <td className={`text-center py-2 font-mono ${book.bookmaker === bestOver.bookmaker ? 'text-emerald-400 font-bold' : 'text-gray-300'}`}>
                {book.overOdds > 0 ? '+' : ''}{book.overOdds}
                {book.bookmaker === bestOver.bookmaker && <span className="ml-1 text-xs">★</span>}
              </td>
              <td className={`text-center py-2 font-mono ${book.bookmaker === bestUnder.bookmaker ? 'text-emerald-400 font-bold' : 'text-gray-300'}`}>
                {book.underOdds > 0 ? '+' : ''}{book.underOdds}
                {book.bookmaker === bestUnder.bookmaker && <span className="ml-1 text-xs">★</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-gray-500 text-xs mt-2">★ = Best available odds</p>
    </div>
  );
};

const AnalysisCard = ({ result }) => {
  const { player, gameData, line, oddsData, bestOver, bestUnder, overAnalysis, underAnalysis, recommendation, bestAnalysis } = result;
  const isPositiveEV = bestAnalysis.ev > 0;
  
  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-bold text-white">{player.name}</h3>
          <p className="text-gray-400 text-sm">{player.teamFull} • {player.position}</p>
        </div>
        <div className={`px-4 py-2 rounded-lg ${isPositiveEV ? 'bg-emerald-500/20 border border-emerald-500/50' : 'bg-red-500/20 border border-red-500/50'}`}>
          <span className={`font-bold ${isPositiveEV ? 'text-emerald-400' : 'text-red-400'}`}>
            {isPositiveEV ? '+EV' : '-EV'}
          </span>
        </div>
      </div>
      
      {/* Game Info */}
      <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-xl p-4 border border-blue-700/30">
        <div className="flex items-center justify-between">
          <div>
            <span className="text-gray-400 text-sm">Today's Game</span>
            <p className="text-white font-semibold">{gameData.awayTeam} @ {gameData.homeTeam}</p>
          </div>
          <div className="text-right">
            <span className="text-gray-400 text-sm">Game Time</span>
            <p className="text-white font-semibold">{new Date(gameData.commenceTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</p>
          </div>
        </div>
      </div>
      
      {/* Odds Table */}
      <OddsTable oddsData={oddsData} line={line} bestOver={bestOver} bestUnder={bestUnder} />
      
      {/* Recommendation Box */}
      <div className={`rounded-xl p-4 border ${recommendation === 'over' ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-blue-500/10 border-blue-500/30'}`}>
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-white font-semibold">🎯 Recommended Play</h4>
          <span className={`px-3 py-1 rounded-full text-sm font-bold ${recommendation === 'over' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-blue-500/20 text-blue-400'}`}>
            {recommendation.toUpperCase()} {line}
          </span>
        </div>
        <p className="text-gray-300 text-sm">
          Best odds at <span className="text-white font-semibold">{bestAnalysis.bestBook}</span>: {' '}
          <span className="font-mono text-emerald-400">{bestAnalysis.bestOdds > 0 ? '+' : ''}{bestAnalysis.bestOdds}</span>
        </p>
      </div>
      
      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <span className="text-gray-400 text-xs uppercase tracking-wide">Model Probability</span>
          <p className="text-2xl font-bold text-white">{(bestAnalysis.trueProbability * 100).toFixed(1)}%</p>
        </div>
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <span className="text-gray-400 text-xs uppercase tracking-wide">Edge vs Market</span>
          <p className={`text-2xl font-bold ${bestAnalysis.edge > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {bestAnalysis.edge > 0 ? '+' : ''}{bestAnalysis.edge.toFixed(1)}%
          </p>
        </div>
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <span className="text-gray-400 text-xs uppercase tracking-wide">Expected Value</span>
          <p className={`text-2xl font-bold ${bestAnalysis.ev > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {bestAnalysis.ev > 0 ? '+' : ''}{bestAnalysis.ev.toFixed(2)}%
          </p>
        </div>
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <span className="text-gray-400 text-xs uppercase tracking-wide">Projected Points</span>
          <p className="text-2xl font-bold text-white">{bestAnalysis.projectedPoints.toFixed(1)}</p>
        </div>
      </div>
      
      {/* Stats Section */}
      <div className="bg-gray-800/30 rounded-xl p-4 border border-gray-700/50">
        <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
          <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
          Player Stats
        </h4>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">Season Average</span>
            <span className="text-white font-medium">{player.seasonAvg} PPG</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Last 10 Games Avg</span>
            <span className="text-white font-medium">{bestAnalysis.recentAvg.toFixed(1)} PPG</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">{gameData.isHome ? 'Home' : 'Away'} Average</span>
            <span className="text-white font-medium">{gameData.isHome ? player.homeAvg : player.awayAvg} PPG</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Hit Rate @ {line} ({recommendation})</span>
            <span className={`font-medium ${bestAnalysis.hitRate >= 0.6 ? 'text-emerald-400' : bestAnalysis.hitRate >= 0.4 ? 'text-yellow-400' : 'text-red-400'}`}>
              {(bestAnalysis.hitRate * 100).toFixed(0)}% ({Math.round(bestAnalysis.hitRate * 10)}/10)
            </span>
          </div>
        </div>
      </div>
      
      {/* Last 10 Visual */}
      <div className="bg-gray-800/30 rounded-xl p-4 border border-gray-700/50">
        <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
          <span className="w-2 h-2 bg-purple-400 rounded-full"></span>
          Last 10 Games
        </h4>
        <div className="flex gap-1 items-end h-20">
          {player.last10.map((pts, i) => {
            const height = (pts / Math.max(...player.last10)) * 100;
            const hitLine = recommendation === 'over' ? pts > line : pts < line;
            return (
              <div key={i} className="flex-1 flex flex-col items-center gap-1">
                <div 
                  className={`w-full rounded-t transition-all ${hitLine ? 'bg-emerald-500' : 'bg-red-500'}`}
                  style={{ height: `${height}%` }}
                />
                <span className="text-[10px] text-gray-500">{pts}</span>
              </div>
            );
          })}
        </div>
        <div className="mt-2 flex items-center gap-2 text-xs">
          <div className="h-px flex-1 bg-yellow-500/50"></div>
          <span className="text-yellow-500">Line: {line}</span>
        </div>
      </div>
      
      {/* Betting Metrics */}
      <div className="bg-gray-800/30 rounded-xl p-4 border border-gray-700/50">
        <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
          <span className="w-2 h-2 bg-amber-400 rounded-full"></span>
          Betting Metrics
        </h4>
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <div>
              <span className="text-gray-400 text-sm">Kelly Criterion</span>
              <p className="text-xs text-gray-500">Optimal bet size (full Kelly)</p>
            </div>
            <span className={`font-bold ${bestAnalysis.kellyFraction > 0 ? 'text-emerald-400' : 'text-gray-400'}`}>
              {bestAnalysis.kellyFraction > 0 ? `${(bestAnalysis.kellyFraction * 100).toFixed(1)}%` : 'No bet'}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <div>
              <span className="text-gray-400 text-sm">Half Kelly (Recommended)</span>
              <p className="text-xs text-gray-500">Conservative sizing</p>
            </div>
            <span className={`font-bold ${bestAnalysis.kellyFraction > 0 ? 'text-blue-400' : 'text-gray-400'}`}>
              {bestAnalysis.kellyFraction > 0 ? `${(bestAnalysis.kellyFraction * 50).toFixed(1)}%` : 'No bet'}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <div>
              <span className="text-gray-400 text-sm">Variance</span>
              <p className="text-xs text-gray-500">{bestAnalysis.variance.description}</p>
            </div>
            <span className="font-bold" style={{ color: bestAnalysis.variance.color }}>
              {bestAnalysis.variance.rating}
            </span>
          </div>
        </div>
      </div>
      
      {/* Confidence Rating */}
      <div className={`rounded-xl p-4 border ${
        bestAnalysis.confidence.rating === 'A' ? 'bg-emerald-500/10 border-emerald-500/30' :
        bestAnalysis.confidence.rating === 'B' ? 'bg-green-500/10 border-green-500/30' :
        bestAnalysis.confidence.rating === 'C' ? 'bg-yellow-500/10 border-yellow-500/30' :
        'bg-red-500/10 border-red-500/30'
      }`}>
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-white font-semibold">Confidence Rating</h4>
            <p className="text-gray-400 text-sm">{bestAnalysis.confidence.label}</p>
          </div>
          <div className="text-4xl font-black" style={{ color: bestAnalysis.confidence.color }}>
            {bestAnalysis.confidence.rating}
          </div>
        </div>
      </div>
      
      {/* Summary */}
      <div className="bg-gradient-to-r from-gray-800/50 to-gray-700/30 rounded-xl p-4 border border-gray-600/50">
        <h4 className="text-white font-semibold mb-2">📊 Analysis Summary</h4>
        <p className="text-gray-300 text-sm leading-relaxed">
          {isPositiveEV ? (
            <>
              <span className="text-emerald-400 font-semibold">{recommendation.toUpperCase()} {line}</span> looks like a +EV opportunity. 
              Model projects {player.name} at {bestAnalysis.projectedPoints.toFixed(1)} points, giving this a {(bestAnalysis.trueProbability * 100).toFixed(1)}% chance vs the implied {(bestAnalysis.impliedProb * 100).toFixed(1)}%.
              Best price at <span className="text-white font-semibold">{bestAnalysis.bestBook}</span> ({bestAnalysis.bestOdds > 0 ? '+' : ''}{bestAnalysis.bestOdds}).
              {bestAnalysis.kellyFraction > 0 && ` Suggested stake: ${(bestAnalysis.kellyFraction * 50).toFixed(1)}% of bankroll.`}
            </>
          ) : (
            <>
              No strong edge found on this prop. Model projects {player.name} at {bestAnalysis.projectedPoints.toFixed(1)} points.
              The {recommendation} has a {(bestAnalysis.trueProbability * 100).toFixed(1)}% chance vs the implied {(bestAnalysis.impliedProb * 100).toFixed(1)}%.
              Consider looking for better value elsewhere.
            </>
          )}
        </p>
      </div>
    </div>
  );
};

const Message = ({ message, isUser }) => {
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[90%] ${isUser ? 'order-2' : 'order-1'}`}>
        {!isUser && (
          <div className="flex items-center gap-2 mb-1">
            <div className="w-6 h-6 rounded-full bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center">
              <span className="text-xs">🎯</span>
            </div>
            <span className="text-gray-400 text-sm font-medium">PropBot</span>
          </div>
        )}
        <div className={`rounded-2xl px-4 py-3 ${
          isUser 
            ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white' 
            : 'bg-gray-800/80 border border-gray-700'
        }`}>
          {message.type === 'text' && (
            <p className={`${isUser ? 'text-white' : 'text-gray-200'} whitespace-pre-wrap`}>{message.content}</p>
          )}
          {message.type === 'analysis' && <AnalysisCard result={message.content} />}
          {message.type === 'error' && (
            <p className="text-red-400 whitespace-pre-wrap">{message.content}</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default function NBAPropsAnalyzer() {
  const [messages, setMessages] = useState([
    {
      isUser: false,
      message: {
        type: 'text',
        content: `Welcome to PropBot! 🏀

I analyze NBA player point props by:
• Fetching live odds from multiple sportsbooks
• Finding the best available lines
• Calculating true probability & expected value
• Recommending the optimal play

Just type a player's name followed by "points":

"Tyrese Haliburton points"
"Luka Doncic points"
"Stephen Curry points"`
      }
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const processInput = (text) => {
    if (!text.trim()) return;
    
    const userInput = text.trim();
    
    // Add user message
    setMessages(prev => [...prev, { isUser: true, message: { type: 'text', content: userInput } }]);
    setIsLoading(true);
    
    // Simulate API delay
    setTimeout(() => {
      const result = analyzeProps(userInput);
      
      if (result.type === 'error') {
        setMessages(prev => [...prev, { isUser: false, message: { type: 'error', content: result.message } }]);
      } else {
        setMessages(prev => [...prev, { isUser: false, message: { type: 'analysis', content: result } }]);
      }
      setIsLoading(false);
    }, 800);
    
    setInput('');
  };
  
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      processInput(input);
    }
  };
  
  const quickPicks = [
    "Luka Doncic points",
    "Shai Gilgeous-Alexander points",
    "LeBron James points"
  ];
  
  return (
    <div className="min-h-screen bg-[#0a0f1a] flex flex-col" style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}>
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 border-b border-gray-700/50 px-4 py-4">
        <div className="max-w-2xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <span className="text-xl">🎯</span>
            </div>
            <div>
              <h1 className="text-white font-bold text-lg">PropBot</h1>
              <p className="text-gray-400 text-xs">NBA Points Props Analyzer</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
            <span className="text-emerald-400 text-sm">Live Odds</span>
          </div>
        </div>
      </div>
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-2xl mx-auto">
          {messages.map((msg, i) => (
            <Message key={i} message={msg.message} isUser={msg.isUser} />
          ))}
          {isLoading && (
            <div className="flex justify-start mb-4">
              <div className="bg-gray-800/80 border border-gray-700 rounded-2xl px-4 py-3">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  <span className="text-gray-400 text-sm ml-2">Fetching odds & analyzing...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>
      
      {/* Quick Picks */}
      {messages.length <= 2 && (
        <div className="px-4 pb-2">
          <div className="max-w-2xl mx-auto">
            <p className="text-gray-500 text-xs mb-2">Quick picks:</p>
            <div className="flex flex-wrap gap-2">
              {quickPicks.map((pick, i) => (
                <button
                  key={i}
                  onClick={() => processInput(pick)}
                  className="px-3 py-1.5 bg-gray-800/50 border border-gray-700 rounded-full text-sm text-gray-300 hover:bg-gray-700/50 hover:border-gray-600 transition-all"
                >
                  {pick}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
      
      {/* Input */}
      <div className="bg-gradient-to-t from-gray-900 to-transparent pt-4 pb-6 px-4">
        <div className="max-w-2xl mx-auto">
          <div className="relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter player name: LeBron James points"
              disabled={isLoading}
              className="w-full bg-gray-800/80 border border-gray-600 rounded-2xl px-5 py-4 pr-14 text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500/50 focus:ring-2 focus:ring-emerald-500/20 transition-all disabled:opacity-50"
            />
            <button
              type="button"
              onClick={(e) => { e.preventDefault(); processInput(input); }}
              disabled={isLoading}
              className="absolute right-2 top-1/2 -translate-y-1/2 w-10 h-10 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-xl flex items-center justify-center hover:opacity-90 transition-opacity disabled:opacity-50"
            >
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            </button>
          </div>
          <p className="text-gray-600 text-xs mt-2 text-center">
            Odds data simulated for demo • Real API integration ready
          </p>
        </div>
      </div>
    </div>
  );
}
