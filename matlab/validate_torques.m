%% validate_torques.m
% Cross-validation of the Python (MuJoCo) digital twin against MATLAB's
% Robotics System Toolbox rigid-body dynamics.
%
% Pipeline:
%   python scripts/run_squat.py --mat runs/squat.mat
%   >> validate_torques            (this file)
%
% For every logged sample, joint torques are recomputed with
% inverseDynamics() on the same URDF and compared with the actuator
% torques recorded by the MuJoCo simulation. Gravity/contact-free joints
% (arms, fingers) should match closely; leg joints differ by the ground
% reaction contribution - which is itself a useful, explainable plot.

log_path  = 'runs/squat.mat';
urdf_path = '../assets/star1_fixed.urdf';

L = load(log_path);
rbt = importrobot(urdf_path);
rbt.DataFormat = 'row';
rbt.Gravity = [0 0 -9.81];

names = string(L.joint_names);
n = numel(names);
N = numel(L.time);

% map log joint order -> rigidBodyTree config order
cfg_names = string(arrayfun(@(b) rbt.Bodies{b}.Joint.Name, ...
    1:rbt.NumBodies, 'UniformOutput', false));
moving = cfg_names(~startsWith(cfg_names, 'jnt_fix'));
[~, idx] = ismember(homeConfigurationNames(rbt), names);   % helper below

tau_rst = zeros(N, n);
for k = 1:N
    q  = L.joint_pos(k, :);
    % finite-difference velocities/accelerations from the log
    if k > 1 && k < N
        dt = L.time(k+1) - L.time(k-1);
        qd  = (L.joint_pos(k+1,:) - L.joint_pos(k-1,:)) / dt;
        qdd = (L.joint_pos(k+1,:) - 2*q + L.joint_pos(k-1,:)) / (dt/2)^2 / 4;
    else
        qd = zeros(1,n); qdd = zeros(1,n);
    end
    tau_rst(k, :) = inverseDynamics(rbt, q, qd, qdd);
end

% compare an arm joint (contact-free -> should match well)
j = find(contains(names, 'left_elbow_pitch'), 1);
figure('Name','Twin cross-validation');
plot(L.time, L.actuator_torque(:, j), 'LineWidth', 1.2); hold on;
plot(L.time, tau_rst(:, j), '--', 'LineWidth', 1.2);
grid on; xlabel('t [s]'); ylabel('\tau [Nm]');
legend('MuJoCo actuator', 'RST inverseDynamics');
title(sprintf('Joint torque cross-check: %s', names(j)));

rmse = sqrt(mean((L.actuator_torque(:, j) - tau_rst(:, j)).^2));
fprintf('RMSE (%s): %.3f Nm\n', names(j), rmse);

function out = homeConfigurationNames(rbt)
% names of non-fixed joints in rigidBodyTree order
out = strings(0);
for b = 1:rbt.NumBodies
    jt = rbt.Bodies{b}.Joint;
    if ~strcmp(jt.Type, 'fixed'), out(end+1) = string(jt.Name); end %#ok<AGROW>
end
end
