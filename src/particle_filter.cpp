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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include <float.h>
#include <assert.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position
    // (based on estimates of x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    double std_x     = std[0];
    double std_y     = std[1];
    double std_theta = std[2];

    default_random_engine gen;

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (unsigned i = 0; i < num_particles; i++)
    {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.;

        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and
    // std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    double dt = delta_t;
    double v = velocity;
    double yr = yaw_rate;

    double std_x     = std_pos[0];
    double std_y     = std_pos[1];
    double std_theta = std_pos[2];

    default_random_engine gen;

    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);

    for (auto &p : particles)
    {
        if (fabs(yr) > 0.001)
        {
            p.x += (v / yr) * (sin(p.theta + yr * dt) - sin(p.theta));
            p.y += (v / yr) * (cos(p.theta) - cos(p.theta + yr * dt));
        }
        // if the yaw rate is small, we're pretty much driving in
        // a straight line.  Avoid the divide by zero above.
        else
        {
            p.x += v*dt*cos(p.theta);
            p.y += v*dt*sin(p.theta);
        }
        p.theta += yr * dt;

        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(
    const std::vector<LandmarkObs> &predicted,
    std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (auto &o : observations)
    {
        double best_dist = FLT_MAX;
        unsigned idx = 0;
        // find the nearest neighbor landmark to the current observation.
        for (auto &p : predicted)
        {
            double dx = p.x - o.x;
            double dy = p.y - o.y;
            double dist = sqrt(dx*dx + dy*dy);

            if (dist < best_dist)
            {
                best_dist = dist;
                idx = p.id;
            }
        }

        o.id = idx;
    }
}

static double calcGauss(double x, double y, double mu_x, double mu_y, double std_x, double std_y)
{
    double dx = x - mu_x;
    double dy = y - mu_y;
    return (1.0 / (2.0*M_PI*std_x*std_y)) * exp(-((dx*dx)/(2*std_x*std_x) + (dy*dy)/(2*std_y*std_y)));
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
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    assert(!observations.empty());

    for (auto &p : particles)
    {
        std::vector<LandmarkObs> predicted;
        for (auto &lm : map_landmarks.landmark_list)
        {
            double dx = lm.x_f - p.x;
            double dy = lm.y_f - p.y;
            double dist = sqrt(dx*dx + dy*dy);

            // Only look at landmarks that could
            // have possibly been sensed by the particle.
            if (dist <= sensor_range)
                predicted.push_back({ lm.id_i, lm.x_f, lm.y_f });
        }

        assert(!predicted.empty());

        std::vector<LandmarkObs> mapObs = observations;

        // Translate observation to map coordinates.
        for (auto &o : mapObs)
        {
            double xm = p.x + cos(p.theta) * o.x - sin(p.theta) * o.y;
            double ym = p.y + sin(p.theta) * o.x + cos(p.theta) * o.y;

            o.x = xm;
            o.y = ym;
        }

        dataAssociation(predicted, mapObs);

        // Debugging visualization
        {
            std::vector<int> assocs;
            std::vector<double> sense_x;
            std::vector<double> sense_y;
            for (auto &o : mapObs)
            {
                assocs.push_back(o.id);
                sense_x.push_back(o.x);
                sense_y.push_back(o.y);
            }
            p = SetAssociations(p, assocs, sense_x, sense_y);
        }

        // Calculate new weight by the product of gaussians.
        double w = 1.;
        for (auto &o : mapObs)
        {
            unsigned lm_idx = o.id - 1;
            assert(lm_idx >= 0 && lm_idx < map_landmarks.landmark_list.size());
            double x = o.x;
            double y = o.y;
            double mu_x = map_landmarks.landmark_list[lm_idx].x_f;
            double mu_y = map_landmarks.landmark_list[lm_idx].y_f;
            double std_x = std_landmark[0];
            double std_y = std_landmark[1];
            w *= calcGauss(x, y, mu_x, mu_y, std_x, std_y);
        }

        p.weight = w;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::vector<double> weights;
    for (auto &p : particles)
        weights.push_back(p.weight);

    default_random_engine gen;
    std::discrete_distribution<> d(weights.begin(), weights.end());

    std::vector<Particle> newParticles;
    for (unsigned i = 0; i < particles.size(); i++)
        newParticles.push_back(particles[d(gen)]);

    particles = newParticles;
}

Particle ParticleFilter::SetAssociations(
    Particle particle,
    std::vector<int> associations,
    std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's
    // (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
