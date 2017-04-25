/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    std::default_random_engine gen;
    
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);
    
    for (int i=0; i<num_particles; i++) {
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
        particles[i].weight = 1/num_particles;
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
    
    std::default_random_engine gen;
    
    
    for (int i=0; i<num_particles; i++) {
        
        if (yaw_rate != 0) {
            double x0 = particles[i].x;
            double y0 = particles[i].y;
            double theta0 = particles[i].theta;

            double x_new = (velocity/yaw_rate)*(sin(theta0+yaw_rate*delta_t)-sin(theta0));
            double y_new = (velocity/yaw_rate)*(cos(theta0)-cos(theta0+yaw_rate*delta_t));
            double theta_new = yaw_rate*delta_t;
            
            std::normal_distribution<double> dist_x(x_new, std_pos[0]);
            std::normal_distribution<double> dist_y(y_new, std_pos[1]);
            std::normal_distribution<double> dist_theta(theta_new, std_pos[2]);
            
            particles[i].x = x0 + dist_x(gen);
            particles[i].y = y0 + dist_y(gen);
            particles[i].theta = theta0 + dist_theta(gen);
        }
        else {
            double x0 = particles[i].x;
            double y0 = particles[i].y;
            double theta0 = particles[i].theta;
            
            double x_new = velocity*delta_t*cos(theta0);
            double y_new = velocity*delta_t*sin(theta0);
            double theta_new = yaw_rate*delta_t;
            
            std::normal_distribution<double> dist_x(x_new, std_pos[0]);
            std::normal_distribution<double> dist_y(y_new, std_pos[1]);
            std::normal_distribution<double> dist_theta(theta_new, std_pos[2]);
            
            particles[i].x = x0 + dist_x(gen);
            particles[i].y = y0 + dist_y(gen);
            particles[i].theta = theta0 + dist_theta(gen);
        }
    }
    

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    std::vector<double> dist;
    observations[0] = predicted[0];
    dist[0] = 1e6;
    for (int i=0; i<observations.size(); i++) {
        for (int j=0; j<predicted.size(); j++) {
            
            float current_dist = sqrt(pow(observations[i].x-predicted[j].x,2) + pow(observations[i].y-predicted[j].y,2));
            if (dist[i] > current_dist) {
                dist[i] = current_dist;
                observations[i] = predicted[j];
            } // end if
        } // end for
    } // end for
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
