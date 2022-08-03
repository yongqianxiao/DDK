function plot_result(x_pred, ref_x, mpc_data, datatype, index, is_save)
    global domain_name seq_nearest_idx alg_name root_path sample_interval start_idx timetoc;
    run_steps = size(x_pred, 1);
    x_pred(:, [3, 6]) = x_pred(:, [3, 6]) * pi / 180;
    x_pred(:, 4:5) = x_pred(:, 4:5) / 3.6;
    fontsize = 11;
    % plot x,y
    subplot(4, 2, 1)
    plot(x_pred(:, 1), x_pred(:, 2), 'b-'); hold on;
    plot(ref_x(1:sample_interval:end, 1), ref_x(1:sample_interval:end, 2), 'r--');
    xlabel('$X$ (m)', 'interpreter','latex', 'FontSize',fontsize, 'FontName','times new roman');
    ylabel('$Y$ (m)', 'interpreter','latex', 'FontSize',fontsize, 'FontName','times new roman');
    subplot(4, 2, 2)
    plot(zeros(1, 1), 'b-');hold on;plot(zeros(1, 1), 'r--');
    Leg1 = legend('Deep EDMD-MPC', 'Real trajectory');
    set(Leg1,'Box','off', 'FontName','times new roman'); % 设置legend 透明
    subplot(4, 2, 3)
    plot(start_idx:1:start_idx+run_steps-1, x_pred(:, 1), 'b-');    hold on;
    plot(ref_x(1:sample_interval:end, 1), 'r--');
    ylabel('$X$ (m)', 'interpreter','latex', 'FontSize',fontsize, 'FontName','times new roman');grid on;
    subplot(4, 2, 4)
    plot(start_idx:1:start_idx+run_steps-1, x_pred(:, 2), 'b-');   hold on;
    plot(ref_x(1:sample_interval:end, 2),'r--')
    ylabel('$Y$ (m)', 'interpreter','latex', 'FontSize',fontsize, 'FontName','times new roman');grid on;
    % plot yaw
    subplot(4, 2, 5)
    plot(start_idx:1:start_idx+run_steps-1, x_pred(:, 3), 'b-'); hold on; plot(ref_x(1:sample_interval:end, 3), 'r--')
    ylabel('$\psi$ (deg)', 'interpreter','latex', 'FontSize',fontsize, 'FontName','times new roman');grid on;
    % plot vx
    subplot(4, 2, 6)
    plot(start_idx:1:start_idx+run_steps-1, x_pred(:, 4), 'b-'); hold on;
    plot(ref_x(1:sample_interval:end, 4), 'r--')
    ylabel('$v_x$ (m/s)', 'interpreter','latex', 'FontSize',fontsize, 'FontName','times new roman');grid on;
    % plot vy
    subplot(4, 2, 7)
    plot(start_idx:1:start_idx+run_steps-1, x_pred(:, 5), 'b-'); hold on;
    plot(ref_x(1:sample_interval:end, 5), 'r--')
    ylabel('$v_y$ (m/s)', 'interpreter','latex', 'FontSize',fontsize, 'FontName','times new roman');grid on;
    % plot yaw rate
    subplot(4, 2, 8)
    plot(start_idx:1:start_idx+run_steps-1, x_pred(:, 6), 'b-'); hold on;
    plot(ref_x(1:sample_interval:end, 6), 'r--')
    ylabel('$\dot{\psi}$ (deg/s)', 'interpreter','latex', 'FontSize',fontsize, 'FontName','times new roman');grid on;
    
    set(gcf,'unit','centimeters','position',[10 5 30 30])
    Leg10 = legend('throttle', 'brake');
    set(Leg10,'Box','off', 'FontName','times new roman'); % 设置legend 透明
        sub_dir = sprintf("DDK_%s", alg_name);
        base_path = strcat(root_path,'/result');

    if is_save
        save_path = strcat(base_path, sprintf('/%s_%d.png', datatype, index));
        mkdir(base_path)
        track_error = [];
        [idx, min_dis] = nearestPoint(ref_x, x_pred(end, :), 0);
        temp_ref = ref_x(start_idx:idx, :);
        temp_x_pred = [];
        temp_idx = 1;
        if size(temp_ref, 1) > size(x_pred, 1)
            temp_ref = [];
            for i=1:size(x_pred, 1)-10
                [temp_idx, temp_err] = nearestPoint(ref_x, x_pred(i, :), temp_idx);
                track_error = [track_error, temp_err];
                temp_x_pred = [temp_x_pred; x_pred(i, :)];
                temp_ref = [temp_ref; ref_x(temp_idx, :)];
            end
        else
            for i=1:size(temp_ref, 1)
                [temp_idx, temp_err] = nearestPoint(x_pred, temp_ref(i, :), temp_idx);
                track_error = [track_error, temp_err];
                temp_x_pred = [temp_x_pred; x_pred(temp_idx, :)];
            end
        end
        abs_error = (temp_x_pred - temp_ref);mean_error = mean(abs(abs_error)); max_error = max(abs(abs_error));
        data = [temp_x_pred, temp_ref];

        save_path       % print save path
        sprintf("-------- Lateral error: %.4f", mean(track_error))
        save(replace(save_path, '.png', '.mat'), 'data', 'abs_error', 'mean_error', 'max_error',...
                'timetoc', 'track_error', 'x_pred', 'ref_x', 'mpc_data', 'seq_nearest_idx')
        saveas(gcf, replace(save_path, '.png', '.fig'));
        saveas(gcf, save_path);
        close;
        subplot(3, 1, 1)
        plot(track_error, 'k--');
        Leg1 = legend('lateral-err');
        set(Leg1,'Box','off', 'FontName','times new roman'); % 设置legend 透明
        xlabel('Time '); ylabel('Dis Err(m)');
        grid on;
        subplot(3, 1, 2)
        plot(abs_error(:, 3), 'c-');hold on; plot(abs_error(:, 6), 'r--');
        xlabel('Time Steps '); ylabel('$\psi$,$\dot{\psi}$ ($rad(/s)$)', 'interpreter','latex');
        grid on;
        subplot(3, 1, 3)
        plot(abs_error(:, 4), 'c-');hold on; plot(abs_error(:, 5), 'r--');
        xlabel('Time Steps '); ylabel('$v_x$,$v_y$ Err ($m/s$)', 'interpreter','latex');
        err_path = strcat(base_path, sprintf('/%s_error_%d.png', datatype, index));
        sgtitle("tracking error")
        grid on;
        set(gcf,'unit','centimeters','position',[10 5 30 30])
        saveas(gcf, err_path);
    end
    close;
end