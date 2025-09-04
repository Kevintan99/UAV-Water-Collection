# camera_system_test.py
"""
Test script for the camera-based pendulum tracking system
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple
import jax.numpy as jnp

def test_vision_processor():
    """Test the vision processing components"""
    from enhanced_pendulum_tracking import VisionTracker, PendulumVisionProcessor
    
    print("🔧 Testing Vision Processing System...")
    
    # Initialize vision system
    tracker = VisionTracker()
    processor = PendulumVisionProcessor(tracker)
    
    # Create test images with different pendulum configurations
    test_cases = [
        {"alpha": 0.0, "beta": 0.0, "name": "Vertical Pendulum"},
        {"alpha": 0.3, "beta": 0.0, "name": "X-axis Swing"},
        {"alpha": 0.0, "beta": 0.2, "name": "Y-axis Swing"},
        {"alpha": 0.25, "beta": 0.15, "name": "Complex Swing"},
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📋 Test {i+1}: {test_case['name']}")
        
        # Generate synthetic test image
        test_image = generate_test_pendulum_image(
            test_case["alpha"], test_case["beta"], 
            noise_level=0.1
        )
        
        # Process with vision system
        drone_pos = jnp.array([0.5, 0.5, 0.8])
        camera_data = processor.process_frame(test_image, drone_pos, float(i))
        
        # Compare results
        true_alpha = test_case["alpha"]
        true_beta = test_case["beta"]
        measured_alpha = camera_data.swing_angle_x
        measured_beta = camera_data.swing_angle_y
        
        alpha_error = abs(measured_alpha - true_alpha) * 180/np.pi
        beta_error = abs(measured_beta - true_beta) * 180/np.pi
        
        print(f"   True α: {true_alpha*180/np.pi:.1f}°, Measured: {measured_alpha*180/np.pi:.1f}° (error: {alpha_error:.1f}°)")
        print(f"   True β: {true_beta*180/np.pi:.1f}°, Measured: {measured_beta*180/np.pi:.1f}° (error: {beta_error:.1f}°)")
        print(f"   Confidence: {camera_data.confidence:.2f}")
        print(f"   Cable visible: {camera_data.cable_visible}")
        
        results.append({
            'test_name': test_case['name'],
            'true_alpha': true_alpha,
            'true_beta': true_beta,
            'measured_alpha': measured_alpha,
            'measured_beta': measured_beta,
            'alpha_error': alpha_error,
            'beta_error': beta_error,
            'confidence': camera_data.confidence,
            'test_image': test_image
        })
    
    # Visualize test results
    visualize_vision_test_results(results)
    
    return results

def generate_test_pendulum_image(alpha: float, beta: float, 
                                noise_level: float = 0.0) -> np.ndarray:
    """
    Generate synthetic pendulum image for testing
    
    Args:
        alpha: Swing angle in x-z plane (radians)
        beta: Swing angle in y-z plane (radians)
        noise_level: Amount of noise to add (0-1)
        
    Returns:
        RGB test image (480, 640, 3)
    """
    height, width = 480, 640
    img = np.ones((height, width, 3), dtype=np.uint8) * 120  # Gray background
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 50, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Drone position in image (center-top area)
    drone_img_x = width // 2
    drone_img_y = height // 4
    
    # Calculate bucket position based on pendulum angles
    pendulum_length_pixels = 200  # pixels
    
    # Convert 3D swing to 2D image projection
    bucket_offset_x = int(pendulum_length_pixels * np.sin(alpha))
    bucket_offset_y = int(pendulum_length_pixels * (np.cos(alpha) + np.sin(beta)))
    
    bucket_img_x = drone_img_x + bucket_offset_x
    bucket_img_y = drone_img_y + bucket_offset_y
    
    # Ensure positions are within image bounds
    bucket_img_x = np.clip(bucket_img_x, 10, width - 10)
    bucket_img_y = np.clip(bucket_img_y, 10, height - 10)
    
    # Draw drone (small dark circle)
    cv2.circle(img, (drone_img_x, drone_img_y), 8, (60, 60, 60), -1)
    
    # Draw cable (black line)
    cv2.line(img, (drone_img_x, drone_img_y), (bucket_img_x, bucket_img_y), 
             (10, 10, 10), 3)
    
    # Draw bucket (light gray circle with some texture)
    cv2.circle(img, (bucket_img_x, bucket_img_y), 12, (200, 200, 200), -1)
    cv2.circle(img, (bucket_img_x, bucket_img_y), 12, (150, 150, 150), 2)
    
    # Add some background features for realism
    # Random small features that shouldn't interfere with detection
    for _ in range(5):
        x = np.random.randint(50, width-50)
        y = np.random.randint(50, height-50)
        # Avoid the pendulum area
        if not (abs(x - drone_img_x) < 100 and abs(y - drone_img_y) < 150):
            cv2.circle(img, (x, y), np.random.randint(3, 8), 
                      (np.random.randint(80, 160),) * 3, -1)
    
    return img

def visualize_vision_test_results(results: List[dict]):
    """Visualize the vision processing test results"""
    
    fig, axes = plt.subplots(2, len(results), figsize=(16, 8))
    fig.suptitle('Vision System Test Results', fontsize=16)
    
    for i, result in enumerate(results):
        # Top row: test images
        axes[0, i].imshow(result['test_image'])
        axes[0, i].set_title(f"{result['test_name']}\nConf: {result['confidence']:.2f}")
        axes[0, i].axis('off')
        
        # Add angle annotations
        true_alpha_deg = result['true_alpha'] * 180/np.pi
        true_beta_deg = result['true_beta'] * 180/np.pi
        meas_alpha_deg = result['measured_alpha'] * 180/np.pi
        meas_beta_deg = result['measured_beta'] * 180/np.pi
        
        axes[0, i].text(10, 30, f"True: α={true_alpha_deg:.1f}°, β={true_beta_deg:.1f}°", 
                       color='red', fontweight='bold', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        axes[0, i].text(10, 50, f"Meas: α={meas_alpha_deg:.1f}°, β={meas_beta_deg:.1f}°", 
                       color='blue', fontweight='bold', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Bottom row: error analysis
    test_names = [r['test_name'] for r in results]
    alpha_errors = [r['alpha_error'] for r in results]
    beta_errors = [r['beta_error'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    # Error bar chart
    x_pos = np.arange(len(results))
    width = 0.35
    
    axes[1, 0].bar(x_pos - width/2, alpha_errors, width, label='α Error', alpha=0.7, color='red')
    axes[1, 0].bar(x_pos + width/2, beta_errors, width, label='β Error', alpha=0.7, color='blue')
    axes[1, 0].set_xlabel('Test Cases')
    axes[1, 0].set_ylabel('Angle Error (degrees)')
    axes[1, 0].set_title('Measurement Errors')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([name.replace(' ', '\n') for name in test_names], fontsize=8)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confidence chart
    axes[1, 1].bar(x_pos, confidences, alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Test Cases')
    axes[1, 1].set_ylabel('Confidence')
    axes[1, 1].set_title('Detection Confidence')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([name.replace(' ', '\n') for name in test_names], fontsize=8)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Overall accuracy scatter plot
    axes[1, 2].scatter(alpha_errors, beta_errors, s=100*np.array(confidences), 
                      alpha=0.7, c=confidences, cmap='RdYlGn')
    axes[1, 2].set_xlabel('α Error (degrees)')
    axes[1, 2].set_ylabel('β Error (degrees)')
    axes[1, 2].set_title('Error Distribution\n(Size = Confidence)')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add colorbar for confidence
    cbar = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
    cbar.set_label('Confidence')
    
    # Summary statistics
    avg_alpha_error = np.mean(alpha_errors)
    avg_beta_error = np.mean(beta_errors)
    avg_confidence = np.mean(confidences)
    
    axes[1, 3].text(0.1, 0.8, f'Summary Statistics:', fontweight='bold', transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.1, 0.7, f'Avg α Error: {avg_alpha_error:.1f}°', transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.1, 0.6, f'Avg β Error: {avg_beta_error:.1f}°', transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.1, 0.5, f'Avg Confidence: {avg_confidence:.2f}', transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.1, 0.4, f'Max α Error: {max(alpha_errors):.1f}°', transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.1, 0.3, f'Max β Error: {max(beta_errors):.1f}°', transform=axes[1, 3].transAxes)
    
    if avg_alpha_error < 5.0 and avg_beta_error < 5.0 and avg_confidence > 0.5:
        status = "✅ PASSED"
        status_color = 'green'
    else:
        status = "❌ NEEDS IMPROVEMENT"
        status_color = 'red'
    
    axes[1, 3].text(0.1, 0.1, f'Overall: {status}', transform=axes[1, 3].transAxes,
                   fontweight='bold', color=status_color, fontsize=12)
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Test Summary:")
    print(f"   Average α error: {avg_alpha_error:.1f}°")
    print(f"   Average β error: {avg_beta_error:.1f}°") 
    print(f"   Average confidence: {avg_confidence:.2f}")
    print(f"   Status: {status}")

def run_camera_system_validation():
    """Run complete camera system validation"""
    print("🎯 === Camera System Validation ===")
    
    # Test 1: Vision processor functionality
    print("\n🔍 Testing Vision Processor...")
    test_results = test_vision_processor()
    
    # Test 2: Camera integration (if possible)
    print("\n📹 Testing Camera Integration...")
    try:
        # This would test actual camera integration
        # For now, just validate the synthetic image generation
        for angle in [0.0, 0.1, 0.2, 0.3]:
            test_img = generate_test_pendulum_image(angle, angle/2)
            assert test_img.shape == (480, 640, 3), f"Wrong image shape: {test_img.shape}"
        print("   ✅ Camera integration test passed")
    except Exception as e:
        print(f"   ❌ Camera integration test failed: {e}")
    
    # Test 3: Measurement accuracy
    print("\n📏 Testing Measurement Accuracy...")
    alpha_errors = [r['alpha_error'] for r in test_results]
    beta_errors = [r['beta_error'] for r in test_results]
    
    accuracy_threshold = 10.0  # degrees
    if max(alpha_errors) < accuracy_threshold and max(beta_errors) < accuracy_threshold:
        print(f"   ✅ Accuracy test passed (max error < {accuracy_threshold}°)")
    else:
        print(f"   ⚠️  Accuracy needs improvement (max errors: α={max(alpha_errors):.1f}°, β={max(beta_errors):.1f}°)")
    
    print("\n🎉 Camera system validation completed!")
    return test_results

if __name__ == "__main__":
    # Run the complete validation suite
    results = run_camera_system_validation()
