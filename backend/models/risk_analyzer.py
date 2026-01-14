import numpy as np
from backend.config import Config

class RiskAnalyzer:
    """Analyze crowd data and calculate risk levels"""
    
    def __init__(self):
        self.risk_history = []
        self.alert_active = False
        
    def calculate_risk_level(self, crowd_count, frame_area, movement_data, iot_data=None):
        """
        Calculate overall risk level
        
        Args:
            crowd_count: Number of people detected
            frame_area: Total frame area (width * height)
            movement_data: Movement analysis results
            iot_data: IoT sensor data (optional)
            
        Returns:
            dict: {
                'level': 'LOW' | 'MEDIUM' | 'HIGH',
                'score': float (0-1),
                'factors': dict of contributing factors,
                'alerts': list of alert messages
            }
        """
        # Calculate density (people per unit area)
        density_score = self._calculate_density_score(crowd_count, frame_area)
        
        # Calculate movement risk
        movement_score = self._calculate_movement_score(movement_data)
        
        # Calculate IoT sensor risk (if available)
        iot_score = self._calculate_iot_score(iot_data) if iot_data else 0.0
        
        # Weighted combination
        weights = {
            'density': 0.5,
            'movement': 0.3,
            'iot': 0.2
        }
        
        overall_score = float(
            density_score * weights['density'] +
            movement_score * weights['movement'] +
            iot_score * weights['iot']
        )
        
        # Determine risk level
        if overall_score < Config.LOW_RISK_THRESHOLD:
            risk_level = 'LOW'
        elif overall_score < Config.MEDIUM_RISK_THRESHOLD:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
        
        # Generate alerts
        alerts = self._generate_alerts(risk_level, crowd_count, movement_data)
        
        # Store in history
        risk_result = {
            'level': risk_level,
            'score': overall_score,
            'factors': {
                'density': density_score,
                'movement': movement_score,
                'iot': iot_score
            },
            'alerts': alerts,
            'crowd_count': crowd_count
        }
        
        self.risk_history.append(risk_result)
        if len(self.risk_history) > 100:
            self.risk_history.pop(0)
        
        return risk_result
    
    def _calculate_density_score(self, count, area):
        """Calculate density-based risk score"""
        # Normalize by area (assuming 640x480 as reference)
        reference_area = 640 * 480
        normalized_count = count * (reference_area / area)
        
        # Map to 0-1 scale (assume 100 people in reference area is max)
        density_score = min(normalized_count / 100.0, 1.0)
        
        return density_score
    
    def _calculate_movement_score(self, movement_data):
        """Calculate movement-based risk score"""
        if movement_data is None:
            return 0.0
        
        score = 0.0
        
        # Movement magnitude contribution
        mag = movement_data.get('movement_magnitude', 0)
        score += min(mag / 10.0, 0.5)  # Max 0.5 contribution
        
        # Erratic movements contribution
        erratic = movement_data.get('erratic_movements', 0)
        score += min(erratic / 20.0, 0.3)  # Max 0.3 contribution
        
        # Panic indicators contribution
        panic_count = len(movement_data.get('panic_indicators', []))
        score += panic_count * 0.1  # 0.1 per indicator
        
        return min(score, 1.0)
    
    def _calculate_iot_score(self, iot_data):
        """Calculate IoT sensor-based risk score"""
        score = 0.0
        
        # Temperature contribution
        temp = iot_data.get('temperature', 25)
        if temp > 30:
            score += min((temp - 30) / 10.0, 0.3)
        
        # Noise level contribution
        noise = iot_data.get('noise_level', 50)
        if noise > 70:
            score += min((noise - 70) / 30.0, 0.4)
        
        # Humidity contribution (high humidity = discomfort)
        humidity = iot_data.get('humidity', 50)
        if humidity > 70:
            score += min((humidity - 70) / 30.0, 0.3)
        
        return min(score, 1.0)
    
    def _generate_alerts(self, risk_level, crowd_count, movement_data):
        """Generate alert messages based on risk factors"""
        alerts = []
        
        if risk_level == 'HIGH':
            alerts.append({
                'severity': 'HIGH',
                'message': 'CRITICAL: High risk of stampede detected!',
                'action': 'Immediate intervention required'
            })
        
        if risk_level == 'MEDIUM':
            alerts.append({
                'severity': 'MEDIUM',
                'message': 'WARNING: Elevated crowd stress levels',
                'action': 'Monitor closely and prepare evacuation routes'
            })
        
        # Specific alerts
        if crowd_count > 50:
            alerts.append({
                'severity': 'INFO',
                'message': f'High crowd density: {crowd_count} people detected',
                'action': 'Consider crowd control measures'
            })
        
        panic_indicators = movement_data.get('panic_indicators', []) if movement_data else []
        if 'convergent_escape_behavior' in panic_indicators:
            alerts.append({
                'severity': 'HIGH',
                'message': 'ALERT: Escape behavior detected in crowd',
                'action': 'Clear evacuation paths immediately'
            })
        
        if 'sudden_direction_changes' in panic_indicators:
            alerts.append({
                'severity': 'MEDIUM',
                'message': 'WARNING: Erratic crowd movements detected',
                'action': 'Investigate cause of disturbance'
            })
        
        return alerts
    
    def predict_contagion_spread(self, density_grid, initial_panic_locations):
        """
        Rule-based prediction of panic spread through crowd
        
        Args:
            density_grid: 2D array of crowd density
            initial_panic_locations: List of (x, y) grid coordinates with panic
            
        Returns:
            2D array showing predicted panic spread intensity
        """
        spread_map = np.zeros_like(density_grid, dtype=np.float32)
        
        # Set initial panic locations
        for x, y in initial_panic_locations:
            if 0 <= y < density_grid.shape[0] and 0 <= x < density_grid.shape[1]:
                spread_map[y, x] = 1.0
        
        # Simple contagion model: panic spreads to adjacent high-density areas
        for iteration in range(3):  # 3 iterations of spread
            new_spread = spread_map.copy()
            
            for y in range(density_grid.shape[0]):
                for x in range(density_grid.shape[1]):
                    if spread_map[y, x] > 0:
                        # Spread to neighbors based on density
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < density_grid.shape[0] and 
                                    0 <= nx < density_grid.shape[1]):
                                    # Spread intensity based on density
                                    spread_intensity = density_grid[ny, nx] / (density_grid.max() + 1e-6)
                                    new_spread[ny, nx] = max(
                                        new_spread[ny, nx],
                                        spread_map[y, x] * 0.7 * spread_intensity
                                    )
            
            spread_map = new_spread
        
        return spread_map
    
    def get_risk_trend(self):
        """Get risk trend over recent history"""
        if len(self.risk_history) < 5:
            return 'insufficient_data'
        
        recent_scores = [h['score'] for h in self.risk_history[-10:]]
        
        # Calculate trend
        first_half = np.mean(recent_scores[:len(recent_scores)//2])
        second_half = np.mean(recent_scores[len(recent_scores)//2:])
        
        if second_half > first_half * 1.3:
            return 'increasing'
        elif second_half < first_half * 0.7:
            return 'decreasing'
        else:
            return 'stable'
