#!/usr/bin/env python3

import numpy as np
from deepmd.env import tf
from deepmd.env import default_tf_session_config
from deepmd.common import make_default_mesh
from deepmd.DeepEval import DeepEval
from deepmd.DataModifier import DipoleChargeModifier

from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

class DeepPot (DeepEval) :
    def __init__(self, 
                 model_file, 
                 default_tf_graph = False) :
        DeepEval.__init__(self, model_file, default_tf_graph = default_tf_graph)
        # self.model_file = model_file
        # self.graph = self.load_graph (self.model_file)
        # checkout input/output tensors from graph
        self.t_ntypes = self.graph.get_tensor_by_name ('load/descrpt_attr/ntypes:0')
        self.t_rcut   = self.graph.get_tensor_by_name ('load/descrpt_attr/rcut:0')
        self.t_dfparam= self.graph.get_tensor_by_name ('load/fitting_attr/dfparam:0')
        self.t_daparam= self.graph.get_tensor_by_name ('load/fitting_attr/daparam:0')
        self.t_tmap   = self.graph.get_tensor_by_name ('load/model_attr/tmap:0')
        # inputs
        self.t_coord  = self.graph.get_tensor_by_name ('load/t_coord:0')
        self.t_type   = self.graph.get_tensor_by_name ('load/t_type:0')
        self.t_natoms = self.graph.get_tensor_by_name ('load/t_natoms:0')
        self.t_box    = self.graph.get_tensor_by_name ('load/t_box:0')
        self.t_mesh   = self.graph.get_tensor_by_name ('load/t_mesh:0')
        # outputs
        self.t_energy = self.graph.get_tensor_by_name ('load/o_energy:0')
        self.t_force  = self.graph.get_tensor_by_name ('load/o_force:0')
        self.t_virial = self.graph.get_tensor_by_name ('load/o_virial:0')
        self.t_ae     = self.graph.get_tensor_by_name ('load/o_atom_energy:0')
        self.t_av     = self.graph.get_tensor_by_name ('load/o_atom_virial:0')
        self.t_fparam = None
        self.t_aparam = None
        # check if the graph has fparam
        for op in self.graph.get_operations():
            if op.name == 'load/t_fparam' :
                self.t_fparam = self.graph.get_tensor_by_name ('load/t_fparam:0')
        self.has_fparam = self.t_fparam is not None
        # check if the graph has aparam
        for op in self.graph.get_operations():
            if op.name == 'load/t_aparam' :
                self.t_aparam = self.graph.get_tensor_by_name ('load/t_aparam:0')
        self.has_aparam = self.t_aparam is not None
        # start a tf session associated to the graph
        self.sess = tf.Session (graph = self.graph, config=default_tf_session_config)        
        [self.ntypes, self.rcut, self.dfparam, self.daparam, self.tmap] = self.sess.run([self.t_ntypes, self.t_rcut, self.t_dfparam, self.t_daparam, self.t_tmap])
        self.tmap = self.tmap.decode('UTF-8').split()
        # setup modifier
        try:
            t_modifier_type = self.graph.get_tensor_by_name('load/modifier_attr/type:0')
            self.modifier_type = self.sess.run(t_modifier_type).decode('UTF-8')
        except ValueError:
            self.modifier_type = None
        except KeyError:
            self.modifier_type = None

        if self.modifier_type == 'dipole_charge':
            t_mdl_name = self.graph.get_tensor_by_name('load/modifier_attr/mdl_name:0')
            t_mdl_charge_map = self.graph.get_tensor_by_name('load/modifier_attr/mdl_charge_map:0')
            t_sys_charge_map = self.graph.get_tensor_by_name('load/modifier_attr/sys_charge_map:0')
            t_ewald_h = self.graph.get_tensor_by_name('load/modifier_attr/ewald_h:0')
            t_ewald_beta = self.graph.get_tensor_by_name('load/modifier_attr/ewald_beta:0')
            [mdl_name, mdl_charge_map, sys_charge_map, ewald_h, ewald_beta] = self.sess.run([t_mdl_name, t_mdl_charge_map, t_sys_charge_map, t_ewald_h, t_ewald_beta])
            mdl_charge_map = [int(ii) for ii in mdl_charge_map.decode('UTF-8').split()]
            sys_charge_map = [int(ii) for ii in sys_charge_map.decode('UTF-8').split()]
            self.dm = DipoleChargeModifier(mdl_name, mdl_charge_map, sys_charge_map, ewald_h = ewald_h, ewald_beta = ewald_beta)
        
        # profiler

        self.profile=False

        if self.profile:
            print("profile : ture")
            self.profiler = model_analyzer.Profiler(graph=self.sess.graph)
            self.run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
            # graph view
            self.profile_graph_opts_builder = option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.time_and_memory())
            self.profile_graph_opts_builder.with_timeline_output(timeline_file='./profile/profiler.json')
            # op view
            self.profile_op_opt_builder_1 = option_builder.ProfileOptionBuilder()
            self.profile_op_opt_builder_1.select(['micros','bytes','float_ops','occurrence'])
            self.profile_op_opt_builder_1.order_by('micros')
            self.profile_op_opt_builder_1.with_max_depth(10)

        print("In init model ")
        print("ntypes : ",self.ntypes)
        print("rcut : ",self.rcut)
        print("dfparam : ",self.dfparam)
        print("daparam : ",self.daparam)
        print("tmap : ",self.tmap)
        print("modifier_type : ",self.modifier_type)



    def get_ntypes(self) :
        return self.ntypes

    def get_rcut(self) :
        return self.rcut

    def get_dim_fparam(self) :
        return self.dfparam

    def get_dim_aparam(self) :
        return self.daparam

    def get_type_map(self):
        return self.tmap

    def eval(self,
             coords,
             cells,
             atom_types,
             fparam = None,
             aparam = None,
             atomic = False) :
        if atomic :
            if self.modifier_type is not None:
                raise RuntimeError('modifier does not support atomic modification')
            return self.eval_inner(coords, cells, atom_types, fparam = fparam, aparam = aparam, atomic = atomic)
        else :
            e, f, v = self.eval_inner(coords, cells, atom_types, fparam = fparam, aparam = aparam, atomic = atomic)
            if self.modifier_type is not None:
                me, mf, mv = self.dm.eval(coords, cells, atom_types)
                e += me.reshape(e.shape)
                f += mf.reshape(f.shape)
                v += mv.reshape(v.shape)
            return e, f, v

    def eval_inner(self,
             coords, 
             cells, 
             atom_types, 
             fparam = None, 
             aparam = None, 
             atomic = False) :
        # standarize the shape of inputs
        atom_types = np.array(atom_types, dtype = int).reshape([-1])
        natoms = atom_types.size
        coords = np.reshape(np.array(coords), [-1, natoms * 3])
        nframes = coords.shape[0]
        if cells is None:
            pbc = False
            # make cells to work around the requirement of pbc
            cells = np.tile(np.eye(3), [nframes, 1]).reshape([nframes, 9])
        else:
            pbc = True
            cells = np.array(cells).reshape([nframes, 9])
        
        if self.has_fparam :
            assert(fparam is not None)
            fparam = np.array(fparam)
        if self.has_aparam :
            assert(aparam is not None)
            aparam = np.array(aparam)

        # reshape the inputs 
        if self.has_fparam :
            fdim = self.get_dim_fparam()
            if fparam.size == nframes * fdim :
                fparam = np.reshape(fparam, [nframes, fdim])
            elif fparam.size == fdim :
                fparam = np.tile(fparam.reshape([-1]), [nframes, 1])
            else :
                raise RuntimeError('got wrong size of frame param, should be either %d x %d or %d' % (nframes, fdim, fdim))
        if self.has_aparam :
            fdim = self.get_dim_aparam()
            if aparam.size == nframes * natoms * fdim:
                aparam = np.reshape(aparam, [nframes, natoms * fdim])
            elif aparam.size == natoms * fdim :
                aparam = np.tile(aparam.reshape([-1]), [nframes, 1])
            elif aparam.size == fdim :
                aparam = np.tile(aparam.reshape([-1]), [nframes, natoms])
            else :
                raise RuntimeError('got wrong size of frame param, should be either %d x %d x %d or %d x %d or %d' % (nframes, natoms, fdim, natoms, fdim, fdim))

        # sort inputs
        coords, atom_types, imap = self.sort_input(coords, atom_types)

        # make natoms_vec and default_mesh
        natoms_vec = self.make_natoms_vec(atom_types)
        assert(natoms_vec[0] == natoms)

        # evaluate
        energy = []
        force = []
        virial = []
        ae = []
        av = []
        feed_dict_test = {}
        feed_dict_test[self.t_natoms] = natoms_vec
        feed_dict_test[self.t_type  ] = atom_types
        t_out = [self.t_energy, 
                 self.t_force, 
                 self.t_virial]
        if atomic :
            t_out += [self.t_ae, 
                      self.t_av]
        for ii in range(nframes) :
            feed_dict_test[self.t_coord] = np.reshape(coords[ii:ii+1, :], [-1])
            feed_dict_test[self.t_box  ] = np.reshape(cells [ii:ii+1, :], [-1])
            if pbc:
                feed_dict_test[self.t_mesh ] = make_default_mesh(cells[ii:ii+1, :])
            else:
                feed_dict_test[self.t_mesh ] = np.array([], dtype = np.int32)
            if self.has_fparam:
                feed_dict_test[self.t_fparam] = np.reshape(fparam[ii:ii+1, :], [-1])
            if self.has_aparam:
                feed_dict_test[self.t_aparam] = np.reshape(aparam[ii:ii+1, :], [-1])

            if self.profile:
                v_out = self.sess.run(t_out, feed_dict = feed_dict_test,options=self.run_options, run_metadata=self.run_metadata)
                self.profiler.add_step(step=ii, run_meta=self.run_metadata)

                self.profiler.profile_graph(self.profile_graph_opts_builder.build())
                self.profiler.profile_operations(self.profile_op_opt_builder_1.build())
            else :
                v_out = self.sess.run (t_out, feed_dict = feed_dict_test)

            energy.append(v_out[0])
            force .append(v_out[1])
            virial.append(v_out[2])
            if atomic:
                ae.append(v_out[3])
                av.append(v_out[4])
        
        # reverse map of the outputs
        force  = self.reverse_map(np.reshape(force, [nframes,-1,3]), imap)
        if atomic :
            ae  = self.reverse_map(np.reshape(ae, [nframes,-1,1]), imap)
            av  = self.reverse_map(np.reshape(av, [nframes,-1,9]), imap)

        energy = np.reshape(energy, [nframes, 1])
        force = np.reshape(force, [nframes, natoms, 3])
        virial = np.reshape(virial, [nframes, 9])
        if atomic:
            ae = np.reshape(ae, [nframes, natoms, 1])
            av = np.reshape(av, [nframes, natoms, 9])
            return energy, force, virial, ae, av
        else :
            return energy, force, virial


    def infer(self,t_coord,t_type,t_box,t_mesh,t_natoms):
        nframes = t_coord.shape[0]
        energy = []
        force = []
        # virial = []
        av = []
        feed_dict = {}
        feed_dict[self.t_mesh ] = t_mesh
        feed_dict[self.t_natoms] = t_natoms

        t_out = [self.t_energy, 
                self.t_force, 
                # self.t_virial,
                self.t_av]

        for ii in range(nframes) :
            feed_dict[self.t_coord] = np.reshape(t_coord[ii:ii+1, :], [-1])
            feed_dict[self.t_type  ] = np.reshape(t_type[ii:ii+1, :], [-1])
            feed_dict[self.t_box  ] = np.reshape(t_box [ii:ii+1, :], [-1])
            if self.profile:
                v_out = self.sess.run (t_out, feed_dict = feed_dict,options=self.run_options,run_metadata=self.run_metadata)
                self.profiler.add_step(step=ii, run_meta=self.run_metadata)
                self.profiler.profile_graph(self.profile_graph_opts_builder.build())
                self.profiler.profile_operations(self.profile_op_opt_builder_1.build())
            else:
                v_out = self.sess.run (t_out, feed_dict = feed_dict)
            energy.append(v_out[0])
            force .append(v_out[1])
            # virial.append(v_out[2])
            av.append(v_out[2])

        # reverse map of the outputs
        # force  = self.reverse_map(np.reshape(force, [nframes,-1,3]), imap)

        energy = np.reshape(energy, [nframes, 1])
        force  = np.reshape(force, [nframes,-1,3])
        # virial = np.reshape(virial, [nframes, 9])
        av  = np.reshape(av, [nframes,-1,9])
        return energy, force, av

