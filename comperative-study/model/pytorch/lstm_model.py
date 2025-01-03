import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMNet(nn.Module):
    def __init__(
        self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True,
        addaptadj=True, aptinit=None, in_dim=2, out_dim=12, residual_channels=32,
        dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2,
        blocks=4, layers=2, one_lstm=True
    ):
        super(LSTMNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.one_lstm = one_lstm
        self.hidden = residual_channels
        self.receptive_field = 1
        self.device = device

        if one_lstm:
            # Suppose we now have a 5-layer LSTM, batch_first=True
            self.rnn_all = nn.LSTM(
                input_size=in_dim * num_nodes,
                hidden_size=self.hidden,
                num_layers=5,      # e.g. 5 stacked layers
                dropout=0.5,       # if you want high dropout for stacked LSTM
                batch_first=True
            ).to(device)

            # 2-layer FC + BN, e.g. in your custom module or however you coded it
            # Just assume we have a small MLP that outputs in_dim*num_nodes
            self.fcn_all = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.hidden, self.hidden),
                nn.BatchNorm1d(self.hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden, in_dim * num_nodes)
            ).to(device)

        else:
            self.rnn = [
                nn.LSTM(in_dim, self.hidden, layers, dropout=dropout, batch_first=True).to(device)
                for _ in range(num_nodes)
            ]
            self.fcn = [
                nn.Linear(self.hidden, in_dim).to(device)
                for _ in range(num_nodes)
            ]

        self.timeframes = out_dim + self.receptive_field
        self.out_dim = out_dim

    def forward(self, input):
        # input: [batch, vals, sensors, measurements]
        batch = input.size(0)
        vals = input.size(1)
        sensors = input.size(2)
        in_len = input.size(3)

        x = F.pad(input, (self.receptive_field, 0, 0, 0))  # => [batch, vals, sensors, measurements + receptive_field]
        timeframes = x.size(3)

        if self.one_lstm:
            # -- Encoder --
            h0 = torch.zeros(5, batch, self.hidden, device=self.device)
            c0 = torch.zeros(5, batch, self.hidden, device=self.device)

            # [batch, timeframes, vals*sensors]
            all_sensors_input = x.view(batch, vals * sensors, -1).transpose(1, 2)

            output, (hn, cn) = self.rnn_all(all_sensors_input, (h0, c0))
            # output: [batch, timeframes, hidden]
            # we take the last time step for the initial decoder input
            decoder_inp_hidd = output[:, -1, :].unsqueeze(1)  # [batch, 1, hidden]

            hdec, cdec = hn, cn  # [5, batch, hidden]

            # -- Decoder Loop --
            decoder_output = torch.zeros(batch, self.out_dim, sensors * vals, device=self.device)

            for t in range(self.out_dim):
                # 1) pass [batch,1,hidden] through FC -> [batch,1,vals*sensors]
                fc_out = self._apply_fcn_all(decoder_inp_hidd)  # shape: [batch, 1, 414] if vals*sensors=414

                # 2) one-step decode in LSTM -> [batch,1,hidden]
                decoder_pred_char, (hdec, cdec) = self.rnn_all(fc_out, (hdec, cdec))
                
                # 3) store this step’s FC output in decoder_output
                #    slice is [batch, 1, 414], so we must NOT squeeze(1) here
                #    or we’ll get [batch, 414] vs. [batch, 1, 414].
                # ----------------------------
                # FIX:
                # ----------------------------
                decoder_output[:, t : t+1, :] = fc_out  # <-- CHANGED (removed .squeeze(1))

                # 4) next iteration’s input is the new LSTM output
                decoder_inp_hidd = decoder_pred_char

            # reshape to [batch, out_dim, sensors, 1]
            output = decoder_output[:, :, :sensors].view(batch, self.out_dim, sensors, 1)
            return output


        else:
            # ========== "isolated_sensors" Branch (unchanged, but fix hidden shapes) ==========
            decoder_output = torch.zeros(batch, self.out_dim, sensors, vals, device=self.device)

            # Split x by sensor => chunk along dim=2
            c = torch.chunk(x, sensors, dim=2)
            for idx, chunk in enumerate(c):
                single_sensor_input = chunk.squeeze(dim=2)  # => [batch, vals, timeframes+receptive_field]
                # batch_first => we want [batch, sequence, features=vals]
                single_sensor_input_sw = single_sensor_input.transpose(1, 2)  # => [batch, timeframes, vals]

                # init hidden => [layers, batch, hidden]
                h0 = torch.zeros(self.layers, batch, self.hidden, device=self.device)
                c0 = torch.zeros(self.layers, batch, self.hidden, device=self.device)

                # run RNN
                output, (hn, cn) = self.rnn[idx](single_sensor_input_sw, (h0, c0))
                # output => [batch, timeframes, hidden]
                # pick last time step => [batch, 1, hidden]
                hdec, cdec = hn, cn  # [layers, batch, hidden]

                # decode loop
                for t in range(self.out_dim):
                    decoder_inp_char = self.fcn[idx](decoder_inp_hidd)  

                    decoder_inp_char = decoder_inp_char.unsqueeze(1)   # -> [64, 1, 414]

                    decoder_pred_char, (hdec, cdec) = self.rnn[idx](decoder_inp_char, (hdec, cdec))
                    decoder_output[:, t : t+1, idx : idx+1, :] = decoder_inp_char
                    decoder_inp_hidd = decoder_pred_char

            # => [batch, out_dim, sensors, 1]
            output = decoder_output[:, :, :, :1].view(batch, self.out_dim, sensors, 1)
            return output

    def _apply_fcn_all(self, decoder_inp_hidd):
        """
        decoder_inp_hidd: [batch, 1, hidden]
        We need [batch, hidden] for BatchNorm1d -> transform -> back to [batch, 1, ...]
        """
        b, seq, hidden_dim = decoder_inp_hidd.shape
        x = decoder_inp_hidd.view(b, hidden_dim)       # [batch, hidden]
        x = self.fcn_all(x)                            # [batch, vals*sensors]
        x = x.unsqueeze(1)                             # [batch, 1, vals*sensors]
        return x


    @classmethod
    def from_args(cls, args, device, supports, aptinit, **kwargs):
        defaults = dict(
            dropout=args.dropout,
            supports=supports,
            addaptadj=args.addaptadj,
            aptinit=aptinit,
            in_dim=args.in_dim,
            out_dim=args.seq_length,
            residual_channels=args.nhid,
            dilation_channels=args.nhid,
            one_lstm=not args.isolated_sensors
        )
        defaults.update(**kwargs)
        model = cls(device, args.num_sensors, **defaults)
        return model

    def load_checkpoint(self, state_dict):
        # only weights that do *NOT* depend on seq_length
        self.load_state_dict(state_dict, strict=False)

