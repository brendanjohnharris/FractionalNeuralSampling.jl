module MakieCoreExt
using MakieCore

# @recipe(MyPlot, x, y, z) do scene
#     Theme()
# end
# function Makie.plot!(myplot::MyPlot)
#     valid_attributes = Makie.shared_attributes(plot, BarPlot)
#     lines!(myplot, rand(10), color = myplot.plot_color)
#     plot!(myplot, myplot.x, myplot.y)
#     myplot
# end
# const MyVolume = MyPlot{Tuple{<:AbstractArray{<:AbstractFloat, 3}}}
# argument_names(::Type{<:MyVolume}) = (:volume,) # again, optional
# function plot!(plot::MyVolume)
#     # plot a volume with a colormap going from fully transparent to plot_color
#     volume!(plot, plot[:volume], colormap = :transparent => plot[:plot_color])
#     plot
# end


end # module
