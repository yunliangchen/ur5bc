#pragma once
#ifndef UR5_UR5_ROBOT_HPP
#define UR5_UR5_ROBOT_HPP

#include <Eigen/Dense>
#include <cmath>
#include <array>
#include <random>
#include <ostream>
#include <iostream> // TODO: remove

namespace ur5 {

    // This model uses parameters consistent with the documentation
    // (not the URDF which is increasingly suspect).
    //
    // https://www.universal-robots.com/how-tos-and-faqs/faq/ur-faq/parameters-for-calculations-of-kinematics-and-dynamics-45257/
    //
    // In particular, it makes use of DH parameters:
    // https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters
    //
    template <class S, int transform = Eigen::Isometry>
    class UR5Robot {
        static_assert(std::is_floating_point_v<S>, "S must be a floating point type");

        static constexpr int NUM_JOINTS = 6;

        static constexpr S PI = 3.1415926535897932384626433832795028841968L;
        
        static constexpr S d1 =  0.089159;
        static constexpr S a2 = -0.42500;
        static constexpr S a3 = -0.39225;
        static constexpr S d4 =  0.10915;
        static constexpr S d5 =  0.09465;
        static constexpr S d6 =  0.08230;

        static constexpr S shoulderOffset_ = 0.13585;
        static constexpr S elbowOffset_ = -0.1197;

        static constexpr S baseLinkRadius_ = 0.073633;
        static constexpr S baseLinkHeight_ = 0.146636; // 0.021 + 0.133651;

        // static constexpr S shoulderLinkRadius_ = 0.059746;
        // static constexpr S shoulderLinkHeight_ = 0.133651;

        static constexpr S shoulderJointRadius_ = 0.059892;
        static constexpr S shoulderJointLength_ = 0.204447 - 0.06982;

        static constexpr S upperArmRadius_ = 0.058697;
        static constexpr S upperArmLength_ = 0.412569;

        static constexpr S elbowRadius_ = 0.059523;
        static constexpr S elbowLength_ = 0.224409; // 0.200904 + 0.023505

        static constexpr S forearmRadius_ = 0.039161;
        static constexpr S forearmLength_ = 0.391457; // 0.333038;
        
        static constexpr S wrist1JointRadius_ = 0.03893;
        static constexpr S wrist1JointLength_ = 0.141719 - 0.02089; // 0.03441 + 0.107309

        static constexpr S wrist1LinkRadius_ = 0.03889;
        static constexpr S wrist1LinkLength_ = 0.150478 - 0.0238;

        static constexpr S wrist2LinkRadius_ = 0.03889;
        static constexpr S wrist2LinkLength_ = 0.122515;
        
    public:
        static constexpr int dimensions = NUM_JOINTS;
        using Scalar = S;
        using Config = Eigen::Matrix<S, NUM_JOINTS, 1>;
        using Frame = Eigen::Transform<S, 3, transform>;
        using Jacobian = Eigen::Matrix<S, 6, NUM_JOINTS>;

    private:
        using T = Eigen::Translation<S, 3>;

        static bool validFrame(const Frame& t) {
            // check that the transform has a valid rotation:
            //    R^T R  = I
            //    det(R) = 1
            if ((t.linear().transpose() * t.linear() - Eigen::Matrix<S,3,3>::Identity()).squaredNorm() > 1e-7) {
                std::clog << "NOT A ROTATION" << std::endl;
                return false;
            }
            
            if (std::abs(1 - t.linear().determinant()) > 1e-5) {
                std::clog << "NOT RIGHT HANDED: " << t.linear().determinant() << std::endl;
                return false;
            }

            // check that the translation is finite:
            if (!t.translation().allFinite()) {
                std::clog << "NON-FINITE TRANSLATION" << std::endl;
                return false;
            }

            return (t.matrix().rows() < 4) ||
                (t(3,0) == 0 && t(3,1) == 0 && t(3,2) == 0 && t(3,3) == 1);
        }

        // This is the offset to the center of the gripper.  Set to 0 to match UR5's tool point.
        // TODO: make this configurable
        static constexpr S eeOffset_ = 0.129459;
        
        Config config_;

        Frame base_;
        std::array<Frame, NUM_JOINTS> jointOrigins_;
        Frame eeFrame_;

        Frame baseLinkFrame() const {
            return base_ * Eigen::Translation<S, 3>(0, 0, baseLinkHeight_/2);
        }

        Frame shoulderJointFrame() const {
            return jointOrigins_[1] * T(
                0, 0, shoulderJointLength_/2 - 0.0284468 + 0.0748479);
        }

        Frame upperArmLinkFrame() const {
            return jointOrigins_[2] * T(-a2, 0, 0) * Ry(0, -1) * T(0, 0, upperArmLength_/2)
                * T(shoulderOffset_, 0, 0);
        }

        Frame elbowJointFrame() const {
            return jointOrigins_[2] // * T(0, 0, 0.065054);
                * T(0, 0, d4 - shoulderOffset_ - elbowOffset_); // 0.082);
        }

        Frame forearmLinkFrame() const {
            return jointOrigins_[3] * T(-a3, 0, 0) * Ry(0, -1) * T(0, 0, forearmLength_/2)
                * T(-d5, 0, 0);
        }

        Frame wrist1JointFrame() const {
            return jointOrigins_[3]
                * T(0, 0, -d6);
        }

        Frame wrist1LinkFrame() const {
            return jointOrigins_[4]
                * T(0, 0, -wrist1LinkLength_/2); // 0.07);
        }

        Frame wrist2LinkFrame() const {
            return jointOrigins_[5] // * Ry(0, -1)
                * T(0, 0, -wrist2LinkLength_/2);
        }
        
    public:
        explicit UR5Robot(const Frame& base = Frame::Identity())
            : base_(base)
        {
        }

        explicit UR5Robot(const Config& q, const Frame& base = Frame::Identity())
            : base_(base)
        {
            setConfig(q);
        }

        const Config& getConfig() const {
            return config_;
        }

        const Frame& eeFrame() const {
            return eeFrame_;
        }

        const Frame& jointOrigin(int i) const {
            return jointOrigins_[i];
        }

    private:
        static Frame Rx(S c, S s) {
            Frame tf;
            tf.setIdentity();
            tf(1,1) = c; tf(1,2) = -s;
            tf(2,1) = s; tf(2,2) = c;
            // std::clog << "Rx(" << c << ", " << s << ") =\n" << tf.matrix() << std::endl;
            return tf;
        }

        static Frame Ry(S c, S s) {
            Frame tf;
            tf.setIdentity();
            tf(0,0) = c;  tf(0,2) = s;
            tf(2,0) = -s; tf(2,2) = c;
            return tf;
        }
        
        static Frame Rz(S c, S s) {
            Frame tf;
            tf.setIdentity();
            tf(0,0) = c; tf(0,1) = -s;
            tf(1,0) = s; tf(1,1) = c;
            return tf;
        }            

        static Frame Rx(S a) {
            return Rx(std::cos(a), std::sin(a));
        }

        static Frame Ry(S a) {
            return Ry(std::cos(a), std::sin(a));
        }

        static Frame Rz(S a) {
            return Rz(std::cos(a), std::sin(a));
        }


    public:
        // the UR5 has a documented limit of 5 kg.
        static constexpr S massLimit_ = 5;
        // The first 3 joints have the same motor and the last 3
        // joints have the same motor, maybe?  (This is from the URDF)
        static constexpr S effortMaxArm_ = 150;
        static constexpr S effortMaxWrist_ = 28;

        static const Config& lowerLimits() {
            // the second joint is limited to -pi/2 to 0 in order to
            // prevent collision between the arm and the table on
            // which it is mounted.
            static constexpr Scalar b = 2*3.1415926535897932384626433832795028841968L;
            static Config min = (Config() << -b, -b/2, -b, -b, -b, -b).finished();
            return min;
        }

        static const Config& upperLimits() {
            static constexpr Scalar b = 2*3.1415926535897932384626433832795028841968L;
            // TODO: second value should maybe be 0
            static Config max = (Config() << b, 0, b, b, b, b).finished();
            return max;
        }

        static const Config& qMin() { return lowerLimits(); }
        static const Config& qMax() { return upperLimits(); }

        static const Config& vMax() {
            static Config v = (Config() << 3, 3, 3, 3, 3, 3).finished();
            return v;
        }

        static const Config& aMax() {
            static Config v{
                (Config() <<
                 effortMaxArm_ / (massLimit_ * 1.25), // shoulder pan
                 effortMaxArm_ / (massLimit_ * 1.25), // shoulder lift
                 effortMaxArm_ / (massLimit_ * 0.75), // elbow
                 effortMaxWrist_ / (massLimit_ * 0.4), // wrist 1
                 effortMaxWrist_ / (massLimit_ * 0.3), // wrist 2
                 effortMaxWrist_ / (massLimit_ * 0.2)).finished()}; // wrist 3
            return v;
        }

        static const Config& jMax() {
            static Config v = aMax() * 10;
            return v;                 
        }

        static const Config& zeroConfig() {
            static Config q = (lowerLimits() + upperLimits()) / 2;
            return q;
        }

        static const Config& configRange() {
            static Config q = upperLimits() - lowerLimits();
            return q;
        }

        template <class RNG>
        static Config randomConfig(RNG& rng) {
            static std::uniform_real_distribution<Scalar> dist;
            Config q;
            for (std::size_t i=0 ; i<NUM_JOINTS ; ++i)
                q[i] = dist(rng);

            return q.cwiseProduct(configRange()) + lowerLimits();
        }

        void setConfig(const Config& q) {
            assert(q.allFinite());

            config_ = q;

            using namespace Eigen;
            using T = Eigen::Translation<S, 3>;
            using V = Matrix<S,3,1>;

            static Frame Ra1 = Rx(0,  1); // also Ra4
            static Frame Ra5 = Rx(0, -1);
            
            jointOrigins_[0] = base_                                           * T(0, 0, d1);
            jointOrigins_[1] = jointOrigins_[0] * Rz(q[0])               * Ra1;
            jointOrigins_[2] = jointOrigins_[1] * Rz(q[1]) * T(a2, 0, 0);
            jointOrigins_[3] = jointOrigins_[2] * Rz(q[2]) * T(a3, 0, 0)       * T(0, 0, d4);
            jointOrigins_[4] = jointOrigins_[3] * Rz(q[3])               * Ra1 * T(0, 0, d5);
            jointOrigins_[5] = jointOrigins_[4] * Rz(q[4])               * Ra5 * T(0, 0, d6);
            eeFrame_         = jointOrigins_[5] * Rz(q[5]) * T(0, 0, eeOffset_);
        }

    private:
        template <class Char, class Traits>
        void blenderAddChildConstraint(
            std::basic_ostream<Char, Traits>& out,
            const std::string& indent,
            const std::string& child,
            const std::string& parent) const
        {
            out << indent << child << ".constraints.new(type='CHILD_OF')\n"
                << indent << child << ".constraints['Child Of'].target = " << parent << "\n";
        }

        template <class Char, class Traits>
        void blenderImportCollada(
            std::basic_ostream<Char, Traits>& out,
            const std::string& indent,
            const std::string& mesh,
            const std::string& name,
            S scale = 1) const
        {
            out << indent << name << " = bpy.data.objects.new(\"" << name << "\", None)\n"
                << indent << name << ".rotation_mode = 'AXIS_ANGLE'\n"
                << indent << "bpy.context.collection.objects.link(" << name << ")\n"
                << indent << "bpy.ops.wm.collada_import(filepath=\"" << mesh << "\")\n";
            
            if (scale != 1)
                out << indent << "bpy.ops.transform.resize(value=(" << scale << ", " << scale << ", " << scale << "))\n";
            
            out << indent << "for i in bpy.context.selected_objects:\n";
            blenderAddChildConstraint(out, indent + "    ", "i", name);
        }

        template <class Char, class Traits>
        void blenderSetAngleAxis(
            std::basic_ostream<Char, Traits>& out,
            const std::string& indent,
            const std::string& name,
            S angle, S x, S y, S z) const
        {
            out << indent << name << ".rotation_axis_angle = (" << angle << ", " << x << ", " << y << ", " << z << ")\n";
        }
        
    public:

        // generates a blender python script to load and present the
        // collada visual meshes for the UR5 in its current
        // configuration.  This creates variables for the "empties"
        // that can later be manipulated to reconfigure the robot.
        // See `updateArticulatedBlenderScript`.  The "namePrefix"
        // variable is currently not supported and partially
        // implemented---leave it as the empty string for now.
        template <class Char, class Traits>
        void toArticulatedBlenderScript(
            std::basic_ostream<Char, Traits>& out,
            const std::string& meshPath,
            const std::string& namePrefix = "",
            const std::string& indent = "") const
        {
            blenderImportCollada(out, indent, meshPath + "Base.dae", namePrefix + "base");

            blenderImportCollada(out, indent, meshPath + "Shoulder.dae", namePrefix + "shoulder");
            blenderAddChildConstraint(out, indent, "shoulder", "base");
            out << indent << namePrefix << "shoulder.location = (0, 0, 0.089159)\n";

            blenderImportCollada(out, indent, meshPath + "UpperArm.dae", namePrefix + "upper_arm");
            blenderAddChildConstraint(out, indent, "upper_arm", "shoulder");
            out << indent << namePrefix << "upper_arm.location = (0, 0.13585, 0)\n";

            blenderImportCollada(out, indent, meshPath + "Forearm.dae", namePrefix + "forearm");
            out << indent << namePrefix << "forearm.location = (0, -0.1197, 0.42500)\n";
            blenderAddChildConstraint(out, indent, "forearm", "upper_arm");

            blenderImportCollada(out, indent, meshPath + "Wrist1.dae", namePrefix + "wrist_1");
            blenderAddChildConstraint(out, indent, "wrist_1", "forearm");
            out << indent << namePrefix << "wrist_1.location = (0, 0, 0.39225)\n";

            blenderImportCollada(out, indent, meshPath + "Wrist2.dae", namePrefix + "wrist_2");
            blenderAddChildConstraint(out, indent, "wrist_2", "wrist_1");
            out << indent << namePrefix << "wrist_2.location = (0, 0.09465, 0)\n";

            blenderImportCollada(out, indent, meshPath + "Wrist3.dae", namePrefix + "wrist_3");
            blenderAddChildConstraint(out, indent, "wrist_3", "wrist_2");
            out << indent << namePrefix << "wrist_3.location = (0, 0, 0.09465)\n";

            bool includeGripper = true;

            if (includeGripper) {
                const std::string gripperMeshPath = "/Users/jeffi/projects/robotiq/robotiq_2f_85_gripper_visualization/meshes/visual/";
            
                // gripper_base
                blenderImportCollada(out, indent, gripperMeshPath + "robotiq_arg2f_85_base_link.dae",
                                     namePrefix + "gripper_base", 0.001);
                blenderAddChildConstraint(out, indent, "gripper_base", "wrist_3");
                out << indent << namePrefix << "gripper_base.location = (0, 0.08230, 0)\n";
                blenderSetAngleAxis(out, indent, "gripper_base", -PI/2, 1, 0, 0);
                
                for (int i=0 ; i<2 ; ++i) {
                    bool left = (i==0);
                    std::string prefix = namePrefix + (left ? "left_" : "right_");
                
                    blenderImportCollada(out, indent, gripperMeshPath + "robotiq_arg2f_85_outer_knuckle.dae",
                                         prefix + "outer_knuckle", 0.001);
                    blenderAddChildConstraint(out, indent, prefix + "outer_knuckle", "gripper_base");
                    out << indent << prefix << "outer_knuckle.location = (0, " << (left ? "-0.0306011" : "0.0306011") << ", 0.054904)\n";
                    
                    if (left)
                        blenderSetAngleAxis(out, indent, prefix + "outer_knuckle", PI, 0, 0, 1);
                    
                    blenderImportCollada(out, indent, gripperMeshPath + "robotiq_arg2f_85_inner_knuckle.dae",
                                         prefix + "inner_knuckle", 0.001);
                    blenderAddChildConstraint(out, indent, prefix + "inner_knuckle", "gripper_base");
                    out << indent << prefix << "inner_knuckle.location = (0, " << (left ? "-0.0127" : "0.0127") << ", 0.06142)\n";
                    
                    if (left)
                        blenderSetAngleAxis(out, indent, prefix + "inner_knuckle", PI, 0, 0, 1);
                    
                    blenderImportCollada(out, indent, gripperMeshPath + "robotiq_arg2f_85_outer_finger.dae",
                                         prefix + "outer_finger", 0.001);
                    blenderAddChildConstraint(out, indent, prefix + "outer_finger", prefix + "outer_knuckle");
                    out << indent << prefix << "outer_finger.location = (0, 0.0315, -0.0041)\n";
                    
                    // axis = X
                    
                    blenderImportCollada(out, indent, gripperMeshPath + "robotiq_arg2f_85_inner_finger.dae",
                                         prefix + "inner_finger", 0.001);
                    blenderAddChildConstraint(out, indent, prefix + "inner_finger", prefix + "outer_finger");
                    out << indent << prefix << "inner_finger.location = (0, 0.0061, 0.0471)\n";
                    
                    blenderImportCollada(out, indent, gripperMeshPath + "robotiq_arg2f_85_pad.dae",
                                         prefix + "inner_finger_pad", 0.001);
                    blenderAddChildConstraint(out, indent, prefix + "inner_finger_pad", prefix + "inner_finger");
                    
                    // URDF has the following, but it seems
                    // unnecessary.  It could be because the URDF does
                    // not use the "...pad.dae" file, and instead uses
                    // a box primitive.
                    // out << indent << prefix << "inner_finger_pad.location = (0, -0.0220203446692936, 0.03242)\n";
                }
            }

            // create the empty for the end effector frame
            out << indent << namePrefix << "ee = bpy.data.objects.new(\"" << namePrefix << "ee\", None)\n"
                << indent << namePrefix << "ee.rotation_mode = 'AXIS_ANGLE'\n"
                << indent << "bpy.context.collection.objects.link(" << namePrefix << "ee)\n";
            blenderAddChildConstraint(out, indent, "ee", "wrist_3");
            out << indent << namePrefix << "ee.rotation_axis_angle = (-" << PI/2 << ", 1, 0, 0)\n";
            out << indent << namePrefix << "ee.location = (0, " << eeOffset_ + d6 << ", 0)\n";


            Eigen::AngleAxis<S> aa;
            aa = eeFrame_.linear();
            
            out << indent << namePrefix << "eeTest = bpy.data.objects.new(\"" << namePrefix << "eeTest\", None)\n"
                << indent << namePrefix << "eeTest.rotation_mode = 'AXIS_ANGLE'\n"
                << indent << "bpy.context.collection.objects.link(" << namePrefix << "eeTest)\n"
                << indent << namePrefix << "eeTest.location = ("
                << eeFrame_.translation()[0] << ", "
                << eeFrame_.translation()[1] << ", "
                << eeFrame_.translation()[2] << ")\n"
                << indent << namePrefix << "eeTest.rotation_axis_angle = ("
                << aa.angle() << ", "
                << aa.axis()[0] << ", "
                << aa.axis()[1] << ", "
                << aa.axis()[2] << ")\n";
            
            out << indent << namePrefix << "ur5_robot = [ "
                << namePrefix << "base, "
                << namePrefix << "shoulder, "
                << namePrefix << "upper_arm, "
                << namePrefix << "forearm, "
                << namePrefix << "wrist_1, "
                << namePrefix << "wrist_2, "
                << namePrefix << "wrist_3 ]\n";

            // TODO: we can remove this and make this method static
            // and have the caller make the
            // updateArticulatedBlenderScript call.
            updateArticulatedBlenderScript(out, namePrefix, indent);
        }

        // updates the blender python script to reconfigure the robot
        // previously rendered by `toArticulatedBlenderScript`.
        template <class Char, class Traits>
        void updateArticulatedBlenderScript(
            std::basic_ostream<Char, Traits>& out,
            const std::string& namePrefix = "",
            const std::string& indent = "") const
        {
            blenderSetAngleAxis(out, indent, "wrist_3", config_[5], 0, 1, 0);
            blenderSetAngleAxis(out, indent, "wrist_2", config_[4], 0, 0, 1);
            blenderSetAngleAxis(out, indent, "wrist_1", config_[3] + PI/2, 0, 1, 0);
            blenderSetAngleAxis(out, indent, "forearm", config_[2], 0, 1, 0);
            blenderSetAngleAxis(out, indent, "upper_arm", config_[1] + PI/2, 0, 1, 0);
            blenderSetAngleAxis(out, indent, "shoulder", config_[0] + PI, 0, 0, 1);
        }
        
        // Computes an analytics inverse kinematic for the UR5.  Due
        // to the UR5's design, there are up to 8 possible
        // configuration that will match the target pose.  We capture
        // each in its own column of the return matrix.  This code
        // assumes that the base frame is I.
        //
        // @param target the target frame for the end effector
        static Eigen::Matrix<S,6,8> ika8(const Frame& target) {
            using namespace Eigen;
            
            assert(validFrame(target));

            // Algorithm credit: original version from:
            // https://github.com/mc-capolei/python-Universal-robot-kinematics/blob/master/universal_robot_kinematics.py
            // then heavily modified and optimized

            Matrix<S, 6, 8> th;
            // th.setZero();

            S d6ee = d6 + eeOffset_;

            // shoulder left/right
            Matrix<S, 3, 1> p05 = target.linear().col(2) * -d6ee + target.translation();
            // std::clog << "p05 = " << p05.transpose() << std::endl;
            // std::clog << "    = " << (target * Matrix<S,3,1>(0,0,-d6ee)).transpose() << std::endl;
            S psi = std::atan2(p05[1], p05[0]);
            S phi = std::acos(d4 / p05.template head<2>().norm());

            th.template block<1,4>(0, 0).fill(PI/2 + psi + phi);
            th.template block<1,4>(0, 4).fill(PI/2 + psi - phi);

            for (int i=0 ; i<8 ; ) {
                Frame t10 = Rx(0,-1) * Rz(-th(0,i)) * T(0, 0, -d1); // * base_.inverse()
                Frame t16 = t10 * target;
                S a = std::acos((t16(2,3) - d4) / d6ee);
                th.template block<1,2>(4, i  ).fill(+a);
                th.template block<1,2>(4, i+2).fill(-a);

                Scalar at61 = std::atan2(-t16(2,1), t16(2,0));
                th.template block<1,2>(5, i  ).fill(at61);
                th.template block<1,2>(5, i+2).fill(at61 < 0 ? at61 + PI : at61 - PI);
                
                for (int j=i+4 ; i<j ; ) {
                    static Frame Td6Rx01 = T(0,0,-d6ee) * Rx(0,1);
                    Frame t14 = t16 * Rz(-th(5,i)) * Td6Rx01 * Rz(-th(4,i)) * T(0,0,-d5);
                     
                    Matrix<S,2,1> p13 = t14.translation().template head<2>()
                        - t14.matrix().template block<2,1>(0,1) * d4;
                    S p13norm = p13.squaredNorm();
                    // S t3 = std::acos(std::complex<S>((p13norm - a2*a2 - a3*a3) / (2 * a2 * a3),0)).real();
                    S t3 = std::acos((p13norm - a2*a2 - a3*a3) / (2 * a2 * a3));
                
                    th(2,i  ) = t3;
                    th(2,i+1) = -t3;

                    Scalar at13 = std::atan2(p13[1], -p13[0]);
                    S asin3 = std::asin(a3 * std::sin(th(2,i))/std::sqrt(p13norm));
                    th(1,i  ) =  asin3 - at13;
                    th(1,i+1) = -asin3 - at13;
                    for (int k=i+2 ; i<k ; ++i) {
                        // th(1,i) = std::asin(a3 * std::sin(th(2,i))/p13norm) - at13;
                        Frame t34 = T(-a3,0,0) * Rz(-th(2,i)) * T(-a2, 0, 0) * Rz(-th(1,i)) * t14;
                        th(3,i) = std::atan2(t34(1,0), t34(0,0));
                    }
                }
            }

            return th;
        }

        // Computes the analytic inverse kinematic solution closest to
        // the robot's current configuration.  Returns true if
        // successful, false if the frame cannot be reached.
        bool ika(const Frame& target) {
            // echo 'scale=40; 8*a(1)' | bc -l
            static constexpr S TAU = 6.2831853071795864769252867665590057683936L;

            Eigen::Matrix<S, 6, 8> all = ika8(target);
            S b = std::numeric_limits<S>::infinity();
            int j = -1;
            for (int i=0 ; i<8 ; ++i) {
                // computes the SO(2) distance between each joint
                Eigen::Array<S, 6, 1> s = (config_ - all.col(i))
                    .cwiseAbs()
                    .unaryExpr([](auto x) { return std::fmod(x, TAU); });
                S d = s.cwiseMin(TAU - s).sum(); // .matrix().template lpNorm<1>();
                if (d < b && !std::isnan(d)) {
                    b = d;
                    j = i;
                }
                // setConfig(all.col(i));
                // std::clog << "=== " << i << " === " << all.col(i).transpose() << std::endl;
                // std::clog << eeFrame().matrix() << std::endl;
            }

            // this can happen if all values are nan
            if (j == -1)
                return false;

            setConfig(all.col(j));
            return true;
        }

    private:
        template <int I>
        void computeJacobianColumn(Jacobian& J) const {
            // auto axis = Eigen::Matrix<S, 3, 1>::UnitZ();
            // Eigen::Matrix<S, 3, 1> m = jointOrigins_[I].linear() * axis;
            // J.template block<3, 1>(3, I) = m;
            J.template block<3, 1>(3, I) = jointOrigins_[I].linear().col(2);
            J.template block<3, 1>(0, I) =
                (jointOrigins_[I].translation() - eeFrame_.translation())
                // .cross(m);
                .cross(jointOrigins_[I].linear().col(2));

            // tail recursion
            if constexpr (I+1 < 6)
                computeJacobianColumn<I+1>(J);
        }
        
    public:
        Jacobian jacobian() const {
            Jacobian J;
            computeJacobianColumn<0>(J);
            return J;
        }
    };
}

#endif
