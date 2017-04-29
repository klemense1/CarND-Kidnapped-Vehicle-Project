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

const size_t NUM_PARTICLES = 100;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Initialize all particles to first position (based on estimates of
    // x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Adds random Gaussian noise to each particle

    std::default_random_engine gen;

    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    for (int i=0; i < NUM_PARTICLES; i++) {
        Particle particle;
        particle.id = -1;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1;
        particles.push_back(particle);
    }

    num_particles = NUM_PARTICLES;
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // Adds measurements to each particle and adds random Gaussian noise

    std::default_random_engine gen;

    double x_new, y_new, theta_new, x0, y0, theta0;

    for (int i=0; i < num_particles; i++) {
        x0 = particles[i].x;
        y0 = particles[i].y;
        theta0 = particles[i].theta;

        if (fabs(yaw_rate) > 1e-6) {
            theta_new = yaw_rate*delta_t;
            x_new = (velocity/yaw_rate) * (sin(theta0 + theta_new) - sin(theta0));
            y_new = (velocity/yaw_rate) * (cos(theta0) - cos(theta0 + theta_new));
        } else {
            x_new = velocity*delta_t*cos(theta0);
            y_new = velocity*delta_t*sin(theta0);
            theta_new = yaw_rate*delta_t;
        }  // end if

        std::normal_distribution<double> dist_x(x_new, std_pos[0]);
        std::normal_distribution<double> dist_y(y_new, std_pos[1]);
        std::normal_distribution<double> dist_theta(theta_new, std_pos[2]);

        particles[i].x = x0 + dist_x(gen);
        particles[i].y = y0 + dist_y(gen);
        particles[i].theta = theta0 + dist_theta(gen);
    }  // end for
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // Finds the predicted measurement that is closest to each observed measurement and assigns the
    //   observed measurement to this particular landmark.

    std::vector<double> distance(observations.size());
    observations[0] = predicted[0];

    for (int i=0; i < observations.size(); ++i) {
        size_t closest_id = -1;

        distance[i] = std::numeric_limits<double>::max();

        for (int j=0; j < predicted.size(); ++j) {
            float current_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

            if (distance[i] > current_dist) {
                distance[i] = current_dist;
                closest_id = j;
            }  // end if
        }  // end for

        observations[i].id = closest_id;
    }  // end for
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // Updates the weights of each particle using a mult-variate Gaussian distribution

    std::vector<LandmarkObs> observations_for_particle(observations.size());

    // transform map landmarks to observation format
    std::vector<LandmarkObs> map_as_observations(map_landmarks.landmark_list.size());

    for (size_t i = 0; i < map_as_observations.size(); ++i) {
        map_as_observations[i].id = 0;
        map_as_observations[i].x = map_landmarks.landmark_list[i].x_f;
        map_as_observations[i].y = map_landmarks.landmark_list[i].y_f;
    }

    // The observations are given in the VEHICLE'S coordinate system. Particles are located
    // according to the MAP'S coordinate system. Transform between the two systems

    for (int i=0; i < num_particles; ++i) {
        for (int j=0; j < observations.size(); ++j) {
            // transformation ... transforming observations into maps coordinate system
            observations_for_particle[j].x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
            observations_for_particle[j].y = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y;
        }  // end for

        // find closest landmark
        dataAssociation(map_as_observations, observations_for_particle);

        for (int j=0; j < observations.size(); ++j) {
            int id = observations_for_particle[j].id;
            double diffx_j = observations_for_particle[j].x - map_as_observations[id].x;
            double diffy_j = observations_for_particle[j].y - map_as_observations[id].y;

            double scaling = 2*M_PI*std_landmark[0]*std_landmark[1];
            double weight_j = exp(-diffx_j*diffx_j/(2*std_landmark[0]*std_landmark[0])-diffy_j*diffy_j/(2*std_landmark[1]*std_landmark[1]))/scaling;

            particles[i].weight *= weight_j;
        }  // end for
    }  // end for
}

void ParticleFilter::resample() {
    // Resampling particles with replacement with probability proportional to their weight.

    static std::default_random_engine gen;

    // Setup the random bits
    double max_weight = 0;
    for (std::vector<Particle>::const_iterator particle = particles.begin(); particle != particles.end(); ++particle) {
        if (particle->weight > max_weight) {
            max_weight = particle->weight;
        }  // end if
    }  // end for

    std::uniform_int_distribution<size_t> uniform_start(0, particles.size() - 1);
    std::uniform_real_distribution<double> uniform_spin(0, 2 * max_weight);

    size_t index = uniform_start(gen);
    double beta = 0;

    std::vector<Particle> new_particles;

    new_particles.reserve(particles.size());

    for (size_t i = 0; i < particles.size(); ++i) {
        beta += uniform_spin(gen);

        while (particles[index].weight < beta) {
            beta -= particles[index].weight;
            index = (index + 1) % particles.size();
        }

        Particle new_particle;
        new_particle.id = -1;
        new_particle.x = particles[index].x;
        new_particle.y = particles[index].y;
        new_particle.theta = particles[index].theta;
        new_particle.weight = 1;
        new_particles.push_back(new_particle);
    }

    particles = new_particles;
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
