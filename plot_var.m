function plot_var(domains)
%PLOT_VAR Plot write_csv domains using a colored scatter heatmap.
% Usage:
%   run('output/p_rank_32_read.m');
%   plot_var(domains);

    if nargin < 1 || isempty(domains)
        error('plot_var requires domains struct from *_read.m');
    end

    figure;
    hold on;
    for k = 1:numel(domains)
        f = domains(k).field;
        x = domains(k).x;
        y = domains(k).y;
        [X, Y] = ndgrid(x, y);
        scatter(X(:), Y(:), 12, f(:), 'filled');
    end
    axis equal;
    colorbar;
    xlabel('x');
    ylabel('y');
    title('Variable field');
    hold off;
end
