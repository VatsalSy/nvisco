from manim import *
import numpy as np
from pylab import cm
import matplotlib
import pickle

config.flush_cache = True
config.disable_caching = True

class scene1(Scene):
    def construct(self):
        title = Tex("Viscoelastic materials").shift(2.5*UP)
        self.play(Write(title))
        # self.wait(0.5)
        heart = SVGMobject("heart.svg").scale(1.2).move_to(5*LEFT)
        hearttitle = Tex("Myocardium").move_to(5*LEFT+1.5*UP).scale(0.7)
        self.play(Write(heart), FadeIn(hearttitle))
        brain = SVGMobject("brain.svg").move_to(1.8*LEFT)
        braintitle = Tex("Brain tissue").move_to(1.8*LEFT+1.5*UP).scale(0.7)
        self.play(Write(brain), FadeIn(braintitle))
        wheel = SVGMobject("wheel.svg").move_to(1.8*RIGHT)
        wheeltitle = Tex("Rubber").move_to(1.8*RIGHT+1.5*UP).scale(0.7)
        self.play(Write(wheel), FadeIn(wheeltitle))
        bc = SVGMobject("bc.svg").move_to(5 *RIGHT)
        bctitle = Tex("Blood clots").move_to(5*RIGHT+1.5*UP).scale(0.7)
        self.play(Write(bc), FadeIn(bctitle))
        self.wait(0.2)

        sample = Rectangle(width=0.5, height=0.5).move_to(heart).shift(0.3*RIGHT+0.5*DOWN).set_color(YELLOW)
        FD = SVGMobject("FD.svg").move_to(sample).scale(0.2)
        self.play(Create(sample))
        self.wait(0.5)

        self.play(sample.animate.move_to(5*RIGHT+DOWN).scale(3),
                  FadeIn(FD),
                  FD.animate.move_to(5*RIGHT+DOWN).scale(3.4),
                  FadeOut(title, heart, hearttitle, brain, braintitle, wheel, wheeltitle, bc, bctitle))
        self.wait()

        MFDarrow = Arrow([0,0,0], [0,1,0]).scale(2.5).next_to(sample,LEFT)
        MFD = Tex("Mean Fiber Direction").next_to(MFDarrow,LEFT).scale(0.7).shift(0.8*RIGHT)
        self.play(Create(MFDarrow), FadeIn(MFD))
        self.wait(0.3)
        self.play(FadeOut(MFD, MFDarrow))

        CFDarrow = Arrow([0,0,0], [1,0,0]).scale(2.5).next_to(sample,UP)
        CFD = Tex("Cross Fiber Direction").next_to(CFDarrow,UP).shift(0.2*LEFT).scale(0.7)
        self.play(Create(CFDarrow), FadeIn(CFD))
        self.wait(0.3)
        self.play(FadeOut(CFD, CFDarrow, FD)) 

        MFDaxes = Axes(
            [0,310], 
            [0,17],
            x_length=6,
            y_axis_config={'include_ticks':False},
            x_axis_config={'include_ticks':False}
            ).scale(0.5).move_to(4*LEFT+DOWN)
        MFDylabel = Tex("Stress").scale(0.7).rotate(90*DEGREES).next_to(MFDaxes,LEFT)
        MFDxlabel = Tex("Time").scale(0.7).next_to(MFDaxes,DOWN)
        MFDtitle = Tex("Mean fiber direction").scale(0.7).next_to(MFDaxes,UP)
         
        CFDaxes = Axes(
            [0,310], 
            [0,17],
            x_length=6,
            y_axis_config={'include_ticks':False},
            x_axis_config={'include_ticks':False}
            ).scale(0.5).move_to(1*RIGHT+DOWN)
        CFDylabel = Tex("Stress").scale(0.7).rotate(90*DEGREES).next_to(CFDaxes,LEFT)
        CFDxlabel = Tex("Time").scale(0.7).next_to(CFDaxes,DOWN)
        CFDtitle = Tex("Cross fiber direction").scale(0.7).next_to(CFDaxes,UP)

        MFDdata = np.genfromtxt("../tikz_data/fig_myocardium1/fig_myocardium1_a.csv")[1:]
        CFDdata = np.genfromtxt("../tikz_data/fig_myocardium1/fig_myocardium1_b.csv")[1:]
        time = MFDdata[:,0]
        MFDstress_gt = MFDdata[:,1]
        MFDstress_nn = MFDdata[:,2]
        CFDstress_gt = CFDdata[:,1]
        CFDstress_nn = CFDdata[:,2]

        MFDpoints = []
        CFDpoints = []
        for t, MFDp, CFDp in zip(time, MFDstress_gt, CFDstress_gt):
            MFDpoints.append(Dot(MFDaxes.coords_to_point(t+5,MFDp+0.5,0)).scale(0.4).set_color(YELLOW))
            CFDpoints.append(Dot(CFDaxes.coords_to_point(t+5,CFDp+0.5,0)).scale(0.4).set_color(YELLOW))

        boundup = Rectangle(width=2, height=0.2).move_to(sample).shift(0.87*UP)
        bounddown = Rectangle(width=2, height=0.2).move_to(sample).shift(0.87*DOWN)
        boundleft = Rectangle(width=0.2, height=2).move_to(sample).shift(0.87*LEFT)
        boundright = Rectangle(width=0.2, height=2).move_to(sample).shift(0.87*RIGHT)
        arrowup = Arrow([0,0,0],[0,1,0]).next_to(boundup, UP).shift(0.2*DOWN)
        arrowdown = Arrow([0,1,0],[0,0,0]).next_to(bounddown, DOWN).shift(0.2*UP)
        arrowleft = Arrow([1,0,0],[0,0,0]).next_to(boundleft, LEFT).shift(0.2*RIGHT)
        arrowright = Arrow([0,0,0],[1,0,0]).next_to(boundright, RIGHT).shift(0.2*LEFT)
        arrowup_label = Tex("Stress").scale(0.5).next_to(arrowup, LEFT)
        arrowleft_label = Tex("Stress").scale(0.5).next_to(arrowleft, UP).shift(0.2*LEFT)

        arrowup_node = arrowup.copy()
        arrowdown_node = arrowdown.copy()
        arrowleft_node = arrowleft.copy()
        arrowright_node = arrowright.copy()
        arrowup_label_node = arrowup_label.copy()
        arrowleft_label_node = arrowleft_label.copy()

        boundup.set_fill(WHITE, opacity=1.0)
        bounddown.set_fill(WHITE, opacity=1.0)
        boundleft.set_fill(WHITE, opacity=1.0)
        boundright.set_fill(WHITE, opacity=1.0)
        experiments = Tex("Experiments show history dependent behavior").move_to(title)
        experiments[0][:11].set_color(YELLOW)
        self.play(Create(MFDaxes), Create(CFDaxes), Create(experiments), FadeIn(MFDylabel, MFDxlabel, CFDylabel, CFDxlabel, MFDtitle, CFDtitle))
        self.wait(0.3)

        sample2 = sample.copy()
        sample3 = sample.copy()
        sample_node = sample.copy()
        sample_tall = Rectangle(height=2.3, width=1.5).set_color(YELLOW).move_to(sample)
        self.add(arrowup, arrowdown, arrowup_label)
        self.play(boundup.animate.shift(0.4*UP), 
                  bounddown.animate.shift(0.4*DOWN), 
                  Transform(sample, sample_tall), 
                  FadeIn(*MFDpoints[:12]),
                  arrowup.animate.shift(0.4*UP), 
                  arrowdown.animate.shift(0.4*DOWN),
                  arrowup_label.animate.shift(0.4*UP))
        for p in MFDpoints[12::2]:
            self.play(FadeIn(p), arrowup.animate.scale(0.97), arrowdown.animate.scale(0.97), run_time=0.05)
        self.remove(sample)
        self.add(sample_tall)

        self.play(FadeOut(boundup, bounddown, arrowup, arrowdown, arrowup_label), run_time=0.4)
        self.play(Transform(sample_tall, sample2), run_time=0.4)
        self.remove(sample_tall)
        self.add(sample2)

        self.play(FadeIn(boundright, boundleft))
        sample_wide = Rectangle(height=1.5, width=2.3).set_color(YELLOW).move_to(sample)
        self.add(arrowleft, arrowright, arrowleft_label)
        self.play(boundleft.animate.shift(0.4*LEFT), 
                  boundright.animate.shift(0.4*RIGHT), 
                  Transform(sample2, sample_wide), 
                  FadeIn(*CFDpoints[:12]),
                  arrowleft.animate.shift(0.4*LEFT), 
                  arrowright.animate.shift(0.4*RIGHT),
                  arrowleft_label.animate.shift(0.4*LEFT))
        for p in CFDpoints[12::2]:
            self.play(FadeIn(p), arrowleft.animate.scale(0.97), arrowright.animate.scale(0.97), run_time=0.05)
        self.remove(sample2)
        self.add(sample_wide)

        self.play(FadeOut(boundleft, boundright, arrowleft, arrowright, arrowleft_label), run_time=0.4)
        self.play(Transform(sample_wide, sample3), run_time=0.4)
        self.remove(sample_wide)
        self.add(sample3)
        self.play(FadeOut(experiments))
 

        #NODE
        nodetitle = Tex(r"Fully data-driven models of finite viscoelasticity \\ with Neural ODEs").move_to(title)
        nodetitle[0][-10:].set_color(BLUE_C)

        sample_node.set_color(BLUE_C)
        sample_node2 = sample_node.copy()
        sample_node3 = sample_node.copy()
        sample_node_tall = Rectangle(height=2.3, width=1.5).set_color(BLUE_C).move_to(sample_node)
        sample_node_wide = Rectangle(height=1.5, width=2.3).set_color(BLUE_C).move_to(sample_node)

        sample_node.shift(3*RIGHT)
        self.play(Write(nodetitle), sample3.animate.shift(5*RIGHT))
        self.play(sample_node.animate.shift(3*LEFT))
        boundup.shift(0.4*DOWN) #Restore to their original positions
        bounddown.shift(0.4*UP)
        boundright.shift(0.4*LEFT)
        boundleft.shift(0.4*RIGHT)
        self.play(FadeIn(boundup, bounddown))
        MFDgraph1 = MFDaxes.plot_line_graph(time[:12]+5, MFDstress_nn[:12]+0.5, add_vertex_dots=False, stroke_width=7).set_color(BLUE_C)
        self.add(arrowup_node, arrowdown_node, arrowup_label_node)
        self.play(boundup.animate.shift(0.4*UP), 
                  bounddown.animate.shift(0.4*DOWN), 
                  Transform(sample_node, sample_node_tall), 
                  Create(MFDgraph1),
                  arrowup_node.animate.shift(0.4*UP), 
                  arrowdown_node.animate.shift(0.4*DOWN),
                  arrowup_label_node.animate.shift(0.4*UP))
        self.remove(sample_node)
        self.add(sample_node_tall)
        MFDgraph2 = MFDaxes.plot_line_graph(time[12:]+5, MFDstress_nn[12:]+0.5, add_vertex_dots=False, stroke_width=7).set_color(BLUE_C)
        self.play(Create(MFDgraph2), arrowup_node.animate.scale(0.3), arrowdown_node.animate.scale(0.3), run_time=3, rate_func=rate_functions.linear)

        self.play(FadeOut(boundup, bounddown, arrowup_node, arrowdown_node, arrowup_label_node), run_time=0.4)
        self.play(Transform(sample_node_tall, sample_node2), run_time=0.4)
        self.remove(sample_node_tall)
        self.add(sample_node2)


        self.play(FadeIn(boundright, boundleft))
        CFDgraph1 = CFDaxes.plot_line_graph(time[:12]+5, CFDstress_nn[:12]+0.5, add_vertex_dots=False, stroke_width=7).set_color(BLUE_C)
        self.add(arrowleft_node, arrowright_node, arrowleft_label_node)
        self.play(boundright.animate.shift(0.4*RIGHT), 
                  boundleft.animate.shift(0.4*LEFT), 
                  Transform(sample_node2, sample_node_wide), 
                  Create(CFDgraph1),
                  arrowleft_node.animate.shift(0.4*LEFT), 
                  arrowright_node.animate.shift(0.4*RIGHT),
                  arrowleft_label_node.animate.shift(0.4*LEFT))
        self.remove(sample_node2)
        self.add(sample_node_wide)
        CFDgraph2 = CFDaxes.plot_line_graph(time[12:]+5, CFDstress_nn[12:]+0.5, add_vertex_dots=False, stroke_width=7).set_color(BLUE_C)
        self.play(Create(CFDgraph2), arrowleft_node.animate.scale(0.3), arrowright_node.animate.scale(0.3), run_time=3, rate_func=rate_functions.linear)

        self.play(FadeOut(boundright, boundleft, arrowleft_node, arrowright_node, arrowleft_label_node), run_time=0.4)
        self.play(Transform(sample_node_wide, sample_node3), run_time=0.4)
        self.remove(sample_node_wide)
        self.add(sample_node3)

        
        
