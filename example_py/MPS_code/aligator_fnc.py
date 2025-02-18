import pinocchio as pin
import numpy as np
import aligator
#from pinocchio.robot_wrapper import RobotWrapper
from aligator import manifolds, dynamics, constraints
import time

def orderAligatorMujoco(data):
    return np.concatenate((data[3:6], data[:3], data[9:12], data[6:9]))

class aligatorControl:
    def __init__(self, model_pin, collision, visual, obstacle, dt, q0, dq0, prev_stage):
        self.model_pin = model_pin
        self.collision = collision
        self.visual = visual
        self.obstacle = obstacle
        self.dt = dt
        self.q0 = q0
        self.data_pin = self.model_pin.createData()
        self.nsteps = 0
        self.prev_stage = prev_stage

        self.jmin = self.model_pin.lowerPositionLimit
        self.jmax = self.model_pin.upperPositionLimit
        self.vmax = self.model_pin.velocityLimit
        self.eff = self.model_pin.effortLimit

        FOOT_FRAME_IDS = {
            fname: self.model_pin.getFrameId(fname) for fname in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        }

        FOOT_JOINT_IDS = {
            fname:  self.model_pin.frames[fid].parentJoint for fname, fid in FOOT_FRAME_IDS.items()
        }
        
        self.nq = self.model_pin.nq
        self.nv = self.model_pin.nv
        self.nu = self.nq - 7
        
        self.x0 = np.concatenate((self.q0, dq0))
        
        pin.forwardKinematics(self.model_pin, self.data_pin, self.q0)
        pin.updateFramePlacements(self.model_pin, self.data_pin)

        self.space = manifolds.MultibodyPhaseSpace(self.model_pin)
        self.ndx = self.space.ndx

        self.actuation_matrix = np.zeros([self.nv, self.nu])
        self.actuation_matrix[6:] = np.eye(self.nu, self.nu)

        # Create dynamics
        self.prox_settings = pin.ProximalSettings(1e-9, 1e-10, 10)
        self.constraint_models = []
        for fname, fid in FOOT_FRAME_IDS.items():
            joint_id = FOOT_JOINT_IDS[fname]
            pl1 = self.model_pin.frames[fid].placement
            pl2 = self.data_pin.oMf[fid]
            cm = pin.RigidConstraintModel(
                pin.ContactType.CONTACT_3D,
                self.model_pin,
                joint_id,
                pl1,
                0,
                pl2,
                pin.LOCAL_WORLD_ALIGNED,
            )
            self.constraint_models.append(cm)

        self.FR_id = self.model_pin.getFrameId("FR_foot")
        self.FL_id = self.model_pin.getFrameId("FL_foot")
        self.RR_id = self.model_pin.getFrameId("RR_foot")
        self.RL_id = self.model_pin.getFrameId("RL_foot")

        self.FR_placement = self.data_pin.oMf[self.FR_id]
        self.FL_placement = self.data_pin.oMf[self.FL_id]
        self.RR_placement = self.data_pin.oMf[self.RR_id]
        self.RL_placement = self.data_pin.oMf[self.RL_id]
           
        z_RL = self.RL_placement.translation[2]
        z_FL = self.FL_placement.translation[2]
        z_FR = self.FR_placement.translation[2]
        z_RR = self.RR_placement.translation[2]
        floor_opt = [z_RL, z_FR, z_RR, z_FL]
        if prev_stage == "ALL":
            self.floor = max(floor_opt)
        elif prev_stage == "FR_RL_h" or prev_stage == "FR_RL":
            self.floor = max(floor_opt[:2])
        else:
            self.floor = max(floor_opt[2:])

        self.trunk_id = self.model_pin.getFrameId("trunk")
        self.trunk_placement = self.data_pin.oMf[self.trunk_id]
        v_ref = pin.Motion()
        v_ref.np[:] = 0.0
        self.frame_vel_FR = aligator.FrameVelocityResidual(self.ndx, self.nu, self.model_pin, v_ref, self.FR_id, pin.LOCAL)
        self.frame_vel_FL = aligator.FrameVelocityResidual(self.ndx, self.nu, self.model_pin, v_ref, self.FL_id, pin.LOCAL)
        self.frame_vel_RR = aligator.FrameVelocityResidual(self.ndx, self.nu, self.model_pin, v_ref, self.RR_id, pin.LOCAL)
        self.frame_vel_RL = aligator.FrameVelocityResidual(self.ndx, self.nu, self.model_pin, v_ref, self.RL_id, pin.LOCAL)
        self.frame_vel_trunk = aligator.FrameVelocityResidual(self.ndx, self.nu, self.model_pin, v_ref, self.trunk_id, pin.LOCAL)

        neutral2 = np.zeros(self.nq + self.nv)
        jmin2 = np.concatenate((np.zeros(self.nq), -self.vmax), axis=None)
        jmax2 = np.concatenate((np.zeros(self.nq),  self.vmax), axis=None)

        for i in range(self.nq):
            if i < 7:
                neutral2[i] = self.space.neutral()[i]
                jmin2[i] = self.jmin[i]
                jmax2[i] = self.jmax[i]
            else:
                neutral2[i] = ((self.jmin[i] - self.jmax[i])/2) + self.jmax[i]
                jmin2[i] = self.jmin[i] - neutral2[i]
                jmax2[i] = self.jmax[i] - neutral2[i]

        state_fn = aligator.StateErrorResidual(self.space, self.nu, neutral2)

        pos_fn = state_fn[6 : self.nv]
        pos_cstr = constraints.BoxConstraint(jmin2[7 : self.nq], jmax2[7 : self.nq])
        self.jlimits = aligator.StageConstraint(pos_fn, pos_cstr)

        vel_fn = state_fn[self.nv + 6 : 2 * self.nv]
        vel_cstr = constraints.BoxConstraint(jmin2[self.nq + 6 :], jmax2[self.nq + 6 :])
        self.vlimits = aligator.StageConstraint(vel_fn, vel_cstr)

        ctrlfn = aligator.ControlErrorResidual(self.ndx, self.nu)
        box_cstr = constraints.BoxConstraint(-self.eff[6:], self.eff[6:])
        self.elimits = aligator.StageConstraint(ctrlfn, box_cstr)
        
        self.w_x_obst = np.array(
                [
                    0,  0,  10, # Trunk position
                    100,  100,  100, # Trunk orientation

                    20,  1e-2,   1e-2, # Front left leg positions
                    20,  1e-2,   1e-2, # Front right leg positions
                    20,  1e-2,   1e-2, # Rear left leg positions
                    20,  1e-2,   1e-2, # Rear right leg positions

                    0,  0,  0, # Trunk linear velocities
                    0,  0,  0, # Trunk angular velocities
                    1e-1,  1e-3,  1e-2, # Front left leg velocities
                    1e-1,  1e-3,  1e-2, # Front right leg velocities
                    1e-1,  1e-3,  1e-2, # Rear left leg velocities
                    1e-1,  1e-3,  1e-2 # Rear right leg velocities

                ]
            )
        
        self.w_x = np.array(
                [
                     0,    0,  10, # Trunk position
                     0,  5,  100, # Trunk orientation

                    30,  0.5,   3, # Front left leg positions
                    30,  0.5,   3, # Front right leg positions
                    30,  0.5,   3, # Rear left leg positions
                    30,  0.5,   3, # Rear right leg positions

                    0,  0,  0, # Trunk linear velocities
                    0,  0,  0, # Trunk angular velocities
                    1e-1,  1e-3,  1e-2, # Front left leg velocities
                    1e-1,  1e-3,  1e-2, # Front right leg velocities
                    1e-1,  1e-3,  1e-2, # Rear left leg velocities
                    1e-1,  1e-3,  1e-2 # Rear right leg velocities

                ]
            ) 
#############################
    def control_step(self, q0, dq0, prev_stage):
        self.prev_stage = prev_stage
        self.x0 = np.concatenate((q0, dq0))
        pin.forwardKinematics(self.model_pin, self.data_pin, q0)
        pin.updateFramePlacements(self.model_pin, self.data_pin)

        self.FR_placement = self.data_pin.oMf[self.FR_id]
        self.FL_placement = self.data_pin.oMf[self.FL_id]
        self.RR_placement = self.data_pin.oMf[self.RR_id]
        self.RL_placement = self.data_pin.oMf[self.RL_id]

        z_RL = self.RL_placement.translation[2]
        z_FL = self.FL_placement.translation[2]
        z_FR = self.FR_placement.translation[2]
        z_RR = self.RR_placement.translation[2]
        floor_opt = [z_RL, z_FR, z_RR, z_FL]
        if prev_stage == "ALL":
            self.floor = max(floor_opt)
        elif prev_stage == "FR_RL_h" or prev_stage == "FR_RL":
            self.floor = max(floor_opt[:2])
        else:
            self.floor = max(floor_opt[2:])

    def create_dynamics(self, support):
        constraint_models = self.constraint_models

        if support == "FR_RL" or support == "FR_RL_h":
            constraints = [constraint_models[0], constraint_models[3]]
            ode = dynamics.MultibodyConstraintFwdDynamics(self.space, self.actuation_matrix, constraints, self.prox_settings)
        elif support == "FL_RR" or support == "FL_RR_h" or support == "FL_RR_align":
            constraints = [constraint_models[1], constraint_models[2]]
            ode = dynamics.MultibodyConstraintFwdDynamics(self.space, self.actuation_matrix, constraints, self.prox_settings)
        else:
            ode = dynamics.MultibodyConstraintFwdDynamics(self.space, self.actuation_matrix, constraint_models, self.prox_settings)

        dyn_model = dynamics.IntegratorSemiImplEuler(ode, self.dt)
        return dyn_model
    
    def createStage(self, support, prev_support, FR_target, RL_target, FL_target, RR_target, trunk_target, add_obst, first):
        ndx = self.ndx
        nu = self.nu
        model_pin = self.model_pin
        space = self.space
        x0 = np.concatenate((self.state_target,np.zeros(model_pin.nv)))
        obstacle = self.obstacle
        
        frame_cs_FR = aligator.FrameTranslationResidual(ndx, nu, model_pin, FR_target.translation, self.FR_id)
        frame_cs_RL = aligator.FrameTranslationResidual(ndx, nu, model_pin, RL_target.translation, self.RL_id)
        frame_cs_FL = aligator.FrameTranslationResidual(ndx, nu, model_pin, FL_target.translation, self.FL_id)
        frame_cs_RR = aligator.FrameTranslationResidual(ndx, nu, model_pin, RR_target.translation, self.RR_id)
        frame_cs_trunk = aligator.FrameTranslationResidual(ndx, nu, model_pin, trunk_target.translation, self.trunk_id)
        
        costs = aligator.CostStack(space, nu)
        
        
        w_Feet = np.eye(3) * 10000
        w_Trunk = np.zeros((3,3))
        

        # Cost matrix
        if add_obst:
            w_x = self.w_x_obst
        else:
            w_x = self.w_x
        w_Trunk[0][0] = 10000
        #w_Trunk[1][1] = 10000

        w_x = np.diag(w_x)
        w_u = np.eye(nu) * 1e-3
        costs.addCost(aligator.QuadraticStateCost(space, nu, x0, w_x))
        costs.addCost(aligator.QuadraticControlCost(space, nu, w_u))
        costs.addCost(aligator.QuadraticResidualCost(space, frame_cs_trunk, w_Trunk))
        

        if support == "FR_RL" or support == "FR_RL_h":
            costs.addCost(aligator.QuadraticResidualCost(space, frame_cs_FL, w_Feet))
            costs.addCost(aligator.QuadraticResidualCost(space, frame_cs_RR, w_Feet))
        elif support == "FL_RR" or support == "FL_RR_h" or support == "FL_RR_align":
            costs.addCost(aligator.QuadraticResidualCost(space, frame_cs_FR, w_Feet))
            costs.addCost(aligator.QuadraticResidualCost(space, frame_cs_RL, w_Feet))

        
        stm = aligator.StageModel(costs, self.create_dynamics(support))
        if add_obst:
            obst = self.obstacle_z(obstacle)
            stm.addConstraint(aligator.StageConstraint(obst, constraints.NegativeOrthant()))
            
        stm.addConstraint(self.jlimits)
        stm.addConstraint(self.vlimits)
        stm.addConstraint(self.elimits)

        if support == "ALL" and (prev_support == "FR_RL" or prev_support == "FR_RL_h"):
            stm.addConstraint(self.frame_vel_FL, constraints.EqualityConstraintSet())
            stm.addConstraint(self.frame_vel_RR, constraints.EqualityConstraintSet())
        elif support == "ALL" and (prev_support == "FL_RR" or prev_support == "FL_RR_h" or prev_support == "FL_RR_align"):
            stm.addConstraint(self.frame_vel_FR, constraints.EqualityConstraintSet())
            stm.addConstraint(self.frame_vel_RL, constraints.EqualityConstraintSet())

        if support == "ALL" or support == "FR_RL" or support == "FR_RL_h":
            stm.addConstraint(frame_cs_FR, constraints.EqualityConstraintSet())
            stm.addConstraint(frame_cs_RL, constraints.EqualityConstraintSet())

        if support == "ALL" or support == "FL_RR" or support == "FL_RR_h" or support == "FL_RR_align":
            stm.addConstraint(frame_cs_FL, constraints.EqualityConstraintSet())
            stm.addConstraint(frame_cs_RR, constraints.EqualityConstraintSet())

        return stm

    def obstacle_z(self, obstacle):
        p_ref = np.zeros(3)
        frame_fun = aligator.FrameTranslationResidual(self.ndx, self.nu, self.model_pin, p_ref, self.trunk_id)
        A = np.array([[0.0, 0.0, 1.0]])
        b = np.array([[obstacle]])
        frame_fun_z = aligator.LinearFunctionComposition(frame_fun, A, -b)
        return frame_fun_z

    def floor_z(self, frame_id):
        p_ref = np.zeros(3)
        frame_fun = aligator.FrameTranslationResidual(self.ndx, self.nu, self.model_pin, p_ref, frame_id)
        A = np.array([[0.0, 0.0, 1.0]])
        b = np.array([[self.floor]])
        frame_fun_z = aligator.LinearFunctionComposition(frame_fun, -A, b)
        return frame_fun_z
    
    def phaseDefinition(self, T_fs, T_ss, x_fwd, swing_apex, init_i, add_obstacle, remove_obstacle, state_target, align):
        self.state_target = state_target

        # Define contact phases and walk parameters
        FL_placement = self.FL_placement
        FR_placement = self.FR_placement
        RL_placement = self.RL_placement
        RR_placement = self.RR_placement
        trunk_placement = self.trunk_placement

        if align:
            n = 1
        else:
            n = 0
        self.align = align
        contact_definition = (
             ["ALL"] * (T_fs * n) # 5
            + ["FL_RR_align"] * (T_ss * n) # 40
            
            + ["ALL"] * T_fs # 5
            + ["FR_RL_h"] * int(T_ss/2) # 20
            + ["ALL"] * T_fs # 5
            + ["FL_RR"] * T_ss # 40
            
            + ["ALL"] * T_fs # 5
            + ["FR_RL"] * T_ss # 40
            + ["ALL"] * T_fs
            + ["FL_RR"] * T_ss

        )

        mpc_horizon = 25
        init_i_obst = init_i
        self.init_i_obst = init_i_obst
        while init_i >= len(contact_definition):
                init_i -= (len(contact_definition) - ((T_fs * n) + (T_ss * n) + T_fs + int(T_ss/2) + T_fs + T_ss))

        if init_i + mpc_horizon >= len(contact_definition):
                init_i2 = (T_fs * n) + (T_ss * n) + T_fs + int(T_ss/2) + T_fs + T_ss
        else:
                init_i2 = 0

        if init_i + mpc_horizon <= len(contact_definition):
                contact_phases = contact_definition[init_i:init_i + mpc_horizon]
        else:
                init_phase = contact_definition[init_i:]
                contact_phases = np.concatenate((init_phase, contact_definition[init_i2:init_i2 + (mpc_horizon - len(init_phase))]))

        self.contact_phases = contact_phases
        FR_placements = []
        RL_placements = []
        FL_placements = []
        RR_placements = []
        trunk_placements = []
        self.nsteps = len(contact_phases)

        check_str = contact_phases[0]
        start_ts_ss = 0
        i = 0
        while check_str == contact_definition[init_i+i]:
            start_ts_ss += 1
            i += 1
            if init_i + i >= len(contact_definition):
                break

        ts_ss = 0
        if check_str == "FR_RL_h" or check_str == "FL_RR_h":
            ts_ss = int(T_ss / 2) - start_ts_ss
        else:
            ts_ss = T_ss - start_ts_ss

        k = 0
        FL_goal = FL_placement.copy()
        FR_goal = FR_placement.copy()
        RL_goal = RL_placement.copy()
        RR_goal = RR_placement.copy()
        trunk_goal = trunk_placement.copy()
        for cp in contact_phases:
            FL_goal_ant = FL_goal
            FR_goal_ant = FR_goal
            RL_goal_ant = RL_goal
            RR_goal_ant = RR_goal
            trunk_goal_ant = trunk_goal

            if cp == "FR_RL_h" or cp == "FL_RR_h":
                T_ss_stage = T_ss / 2
            else:
                T_ss_stage = T_ss

            if cp == "ALL":
                ts_ss = 0
                FR_placements.append(FR_goal_ant)
                FL_placements.append(FL_goal_ant)
                RR_placements.append(RR_goal_ant)
                RL_placements.append(RL_goal_ant)
                trunk_placements.append(trunk_goal_ant)
            if cp == "FR_RL" or cp == "FR_RL_h":
                ts_ss += 1
                FL_goal = FL_placement.copy()
                RR_goal = RR_placement.copy()
                FR_placements.append(FR_goal_ant)
                RL_placements.append(RL_goal_ant)
                trunk_goal = trunk_placement.copy()

                T_ss_stage_1 = T_ss_stage - (ts_ss - 1)

                x_fwd_stage = (FR_goal_ant.translation[0] + (x_fwd/2)) - FL_goal_ant.translation[0]
                FL_goal.translation[0] = FL_goal_ant.translation[0] + self.xytraj(x_fwd_stage, T_ss_stage_1)
                x_fwd_stage = (RL_goal_ant.translation[0] + (x_fwd/2)) - RR_goal_ant.translation[0]
                RR_goal.translation[0] = RR_goal_ant.translation[0] + self.xytraj(x_fwd_stage, T_ss_stage_1)

                x_dist = RL_goal_ant.translation[0]  + (x_fwd/2) + (FR_goal_ant.translation[0] - RL_goal_ant.translation[0]) / 2
                x_fwd_stage = x_dist - trunk_goal_ant.translation[0]
                trunk_goal.translation[0] = trunk_goal_ant.translation[0] + (self.xytraj(x_fwd_stage, T_ss_stage_1))

                z_coord = self.floor + self.ztraj(swing_apex, T_ss_stage, ts_ss)

                FL_goal.translation[2] = z_coord
                FL_placements.append(FL_goal)
                
                RR_goal.translation[2] = z_coord
                RR_placements.append(RR_goal)
                
                trunk_placements.append(trunk_goal)
            if cp == "FL_RR" or cp == "FL_RR_h" or cp == "FL_RR_align":
                ts_ss += 1
                FR_goal = FR_placement.copy()
                RL_goal = RL_placement.copy()
                FL_placements.append(FL_goal_ant)
                RR_placements.append(RR_goal_ant)

                trunk_goal = trunk_placement.copy()

                T_ss_stage_1 = T_ss_stage - (ts_ss - 1)
                if cp == "FL_RR_align":
                    x_fwd_stage = FL_goal_ant.translation[0] - FR_goal_ant.translation[0]
                    FR_goal.translation[0] = FR_goal_ant.translation[0] + self.xytraj(x_fwd_stage, T_ss_stage_1)
                    x_fwd_stage = RR_goal_ant.translation[0] - RL_goal_ant.translation[0]
                    RL_goal.translation[0] = RL_goal_ant.translation[0] + self.xytraj(x_fwd_stage, T_ss_stage_1)

                    y_fwd_stage = FL_goal_ant.translation[1] - RL_goal_ant.translation[1]
                    RL_goal.translation[1] = RL_goal_ant.translation[1] + self.xytraj(y_fwd_stage, T_ss_stage_1)
                    y_fwd_stage = RR_goal_ant.translation[1] - FR_goal_ant.translation[1]
                    FR_goal.translation[1] = FR_goal_ant.translation[1] + self.xytraj(y_fwd_stage, T_ss_stage_1)

                    x_dist = RR_goal_ant.translation[0] + (FL_goal_ant.translation[0] - RR_goal_ant.translation[0]) / 2
                    x_fwd_stage = x_dist - trunk_goal_ant.translation[0]
                    trunk_goal.translation[0] = trunk_goal_ant.translation[0] + (self.xytraj(x_fwd_stage, T_ss_stage_1))

                    y_dist = FL_goal_ant.translation[1] + (RR_goal_ant.translation[1] - FL_goal_ant.translation[1]) / 2
                    y_fwd_stage = y_dist - trunk_goal_ant.translation[1]
                    trunk_goal.translation[1] = trunk_goal_ant.translation[1] + (self.xytraj(y_fwd_stage, T_ss_stage_1)) 
                    
                else:
                    x_fwd_stage = (FL_goal_ant.translation[0] + (x_fwd/2)) - FR_goal_ant.translation[0]
                    FR_goal.translation[0] = FR_goal_ant.translation[0] + self.xytraj(x_fwd_stage, T_ss_stage_1)
                    x_fwd_stage = (RR_goal_ant.translation[0] + (x_fwd/2)) - RL_goal_ant.translation[0]
                    RL_goal.translation[0] = RL_goal_ant.translation[0] + self.xytraj(x_fwd_stage, T_ss_stage_1)

                    x_dist = RR_goal_ant.translation[0] + (x_fwd/2) + (FL_goal_ant.translation[0] - RR_goal_ant.translation[0]) / 2
                    x_fwd_stage = x_dist - trunk_goal_ant.translation[0]
                    trunk_goal.translation[0] = trunk_goal_ant.translation[0] + (self.xytraj(x_fwd_stage , T_ss_stage_1)) 

                z_coord = self.floor + self.ztraj(swing_apex, T_ss_stage, ts_ss)

                FR_goal.translation[2] = z_coord
                FR_placements.append(FR_goal)
                
                RL_goal.translation[2] = z_coord
                RL_placements.append(RL_goal)

                trunk_placements.append(trunk_goal)
            k += 1
            
        
        for i in range(0,self.nsteps):
            check_i = i+init_i_obst
            if (check_i > 95) and (check_i < 1850) and add_obstacle:
                add = True

                if  check_i > 725 and check_i < 1220:
                    self.obstacle = 0.28
                elif  check_i > 680 and check_i < 1265:
                    self.obstacle = 0.29
                elif  check_i > 635 and check_i < 1310:
                    self.obstacle = 0.30
                elif  check_i > 590 and check_i < 1355:
                    self.obstacle = 0.31
                elif  check_i > 545 and check_i < 1400:
                    self.obstacle = 0.32
                elif  check_i > 500 and check_i < 1445:
                    self.obstacle = 0.33
                elif  check_i > 455 and check_i < 1490:
                    self.obstacle = 0.34
                elif  check_i > 410 and check_i < 1535:
                    self.obstacle = 0.35
                elif  check_i > 365 and check_i < 1580:
                    self.obstacle = 0.36
                elif  check_i > 320 and check_i < 1625:
                    self.obstacle = 0.37
                elif  check_i > 275 and check_i < 1670:
                    self.obstacle = 0.38
                elif  check_i > 230 and check_i < 1715:
                    self.obstacle = 0.39
                elif  check_i > 185 and check_i < 1760:
                    self.obstacle = 0.40
                elif  check_i > 140 and check_i < 1805:
                    self.obstacle = 0.41
                elif  check_i > 95 and check_i < 1850:
                    self.obstacle = 0.42

            else:
                add = False
            
            if i == 0:
                stages = [self.createStage(contact_phases[i], self.prev_stage, FR_placements[i], RL_placements[i], FL_placements[i], RR_placements[i], trunk_placements[i], add, True)]
            else: 
                stages.append(self.createStage(contact_phases[i], contact_phases[i - 1], FR_placements[i], RL_placements[i], FL_placements[i], RR_placements[i], trunk_placements[i], add, False))

        self.add = add
        self.FR_placements_last = FR_placements[-1]
        self.RL_placements_last = RL_placements[-1]
        self.FL_placements_last = FL_placements[-1]
        self.RR_placements_last = RR_placements[-1]
        self.trunk_placements_last = trunk_placements[-1]
        self.trunk_placements_last.translation[2] = state_target[2]

        self.first_contact = contact_phases[0]
        self.second_contact = contact_phases[1]
        self.last_contact = contact_phases[-1]
        self.prev_last_contact = contact_phases[-2]

        self.FR_placements = FR_placements
        self.RL_placements = RL_placements
        self.FL_placements = FL_placements
        self.RR_placements = RR_placements
        return stages
    
    def xytraj(self, x_fwd, t_ss):
        return (x_fwd/t_ss)

    def ztraj(self, swing_apex, t_ss, ts_ss):
        return swing_apex * np.sin(np.pi/t_ss * ts_ss)
    
    def solveAligator(self, initial_state, stages, x_warmstart, u_warmstart):
        start = time.time()
        ndx = self.ndx
        nu = self.nu
        model_pin = self.model_pin
        space = self.space
        
        frame_cs_FR = aligator.FrameTranslationResidual(ndx, nu, model_pin, self.FR_placements_last.translation, self.FR_id)
        frame_cs_RL = aligator.FrameTranslationResidual(ndx, nu, model_pin, self.RL_placements_last.translation, self.RL_id)
        frame_cs_FL = aligator.FrameTranslationResidual(ndx, nu, model_pin, self.FL_placements_last.translation, self.FL_id)
        frame_cs_RR = aligator.FrameTranslationResidual(ndx, nu, model_pin, self.RR_placements_last.translation, self.RR_id)
        frame_cs_trunk = aligator.FrameTranslationResidual(ndx, nu, model_pin, self.trunk_placements_last.translation, self.trunk_id)

        term_costs = aligator.CostStack(space, nu)

        w_Feet = np.eye(3) * 10000
        w_Trunk = np.zeros((3,3))
        

        # Cost matrix
        if self.add:
            w_x = self.w_x_obst
        else:
            w_x = self.w_x
        w_Trunk[0][0] = 10000
        

        x0 = np.concatenate((self.state_target,np.zeros(model_pin.nv)))
        w_x = np.diag(w_x)
        w_u = np.eye(nu) * 1e-3
        term_costs.addCost(aligator.QuadraticStateCost(space, nu, x0, w_x))
        term_costs.addCost(aligator.QuadraticControlCost(space, nu, w_u))
        term_costs.addCost(aligator.QuadraticResidualCost(space, frame_cs_trunk, w_Trunk))
        

        if self.last_contact == "FR_RL" or self.last_contact == "FR_RL_h":
            term_costs.addCost(aligator.QuadraticResidualCost(space, frame_cs_FL, w_Feet))
            term_costs.addCost(aligator.QuadraticResidualCost(space, frame_cs_RR, w_Feet))
        elif self.last_contact == "FL_RR" or self.last_contact == "FL_RR_h" or self.last_contact == "FL_RR_align":
            term_costs.addCost(aligator.QuadraticResidualCost(space, frame_cs_FR, w_Feet))
            term_costs.addCost(aligator.QuadraticResidualCost(space, frame_cs_RL, w_Feet))

        problem = aligator.TrajOptProblem(initial_state, stages, term_costs)

        if self.last_contact == "ALL" or self.last_contact == "FR_RL" or self.last_contact == "FR_RL_h":
            problem.addTerminalConstraint(aligator.StageConstraint(frame_cs_RL, constraints.EqualityConstraintSet()))
            problem.addTerminalConstraint(aligator.StageConstraint(frame_cs_FR, constraints.EqualityConstraintSet()))

        if self.last_contact == "ALL" or self.last_contact == "FL_RR" or self.last_contact == "FL_RR_h" or self.last_contact == "FL_RR_align":
            problem.addTerminalConstraint(aligator.StageConstraint(frame_cs_FL, constraints.EqualityConstraintSet()))
            problem.addTerminalConstraint(aligator.StageConstraint(frame_cs_RR, constraints.EqualityConstraintSet()))

        if self.add:
            obst = self.obstacle_z(self.obstacle)
            problem.addTerminalConstraint(aligator.StageConstraint(obst, constraints.NegativeOrthant()))

        if self.last_contact == "ALL" and (self.prev_last_contact == "FR_RL" or self.prev_last_contact == "FR_RL_h"):
            problem.addTerminalConstraint(aligator.StageConstraint(self.frame_vel_FL, constraints.EqualityConstraintSet()))
            problem.addTerminalConstraint(aligator.StageConstraint(self.frame_vel_RR, constraints.EqualityConstraintSet()))
        elif self.last_contact == "ALL" and (self.prev_last_contact == "FL_RR" or self.prev_last_contact == "FL_RR_h" or self.prev_last_contact == "FL_RR_align"):
            problem.addTerminalConstraint(aligator.StageConstraint(self.frame_vel_FR, constraints.EqualityConstraintSet()))
            problem.addTerminalConstraint(aligator.StageConstraint(self.frame_vel_RL, constraints.EqualityConstraintSet()))
        
        if self.add:
            problem.addTerminalConstraint(aligator.StageConstraint(frame_cs_trunk[:2], constraints.EqualityConstraintSet()))
        else:
            problem.addTerminalConstraint(aligator.StageConstraint(frame_cs_trunk[:3], constraints.EqualityConstraintSet()))

        tol = 1e-2
        mu_init = 1e-4
        solver = aligator.SolverProxDDP(tol, mu_init=mu_init, verbose=aligator.QUIET)#VERBOSE)#
        solver.max_iters = 3
        solver.rollout_type = aligator.ROLLOUT_LINEAR
        solver.linear_solver_choice = aligator.LQ_SOLVER_PARALLEL 
        solver.force_initial_condition = True
        solver.setNumThreads(8)
        start = time.time()
        solver.setup(problem)
        setup_time = time.time() - start
       # print('setup_time', setup_time)

        if np.all(x_warmstart == 0) and np.all(u_warmstart == 0):
            xs_i = [self.x0] * (self.nsteps + 1)
            us_i = [np.zeros(self.nu)] * self.nsteps
            
        else:
            xs_i = x_warmstart
            us_i = u_warmstart
            
        start = time.time()
        solver.run(problem, xs_i, us_i)
        run_time = time.time()-start
       # print('run_time', run_time)

        res = solver.results
        joint_res = [x[:self.nq] for x in res.xs]
        joint_vels = [x[self.nq:] for x in res.xs]
        x_res = [x for x in res.xs]
        eff_res = [x for x in res.us]

        return joint_res, joint_vels, x_res, eff_res, self.first_contact, self.second_contact, res.conv, setup_time, run_time