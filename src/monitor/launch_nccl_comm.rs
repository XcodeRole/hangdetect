use crate::cuda_funcs::cuda_stream_get_id;
use std::ffi::c_void;
use std::fmt::{Display, Formatter};

pub enum NCCLCommunication {
    AllReduce {
        comm: *const c_void,
        stream: *const c_void,
        count: u64,
        datatype: u32,
        op: u32,
    },
    Broadcast {
        comm: *const c_void,
        stream: *const c_void,
        count: u64,
        datatype: u32,
        root: u32,
    },
    Reduce {
        comm: *const c_void,
        stream: *const c_void,
        count: u64,
        datatype: u32,
        op: u32,
        root: u32,
    },
    AllGather {
        comm: *const c_void,
        stream: *const c_void,
        sendcount: u64,
        datatype: u32,
    },
    ReduceScatter {
        comm: *const c_void,
        stream: *const c_void,
        recvcount: u64,
        datatype: u32,
        op: u32,
    },
    // AlltoAll {
    //     comm: *const c_void,
    //     stream: *const c_void,
    //     count: u64,
    //     datatype: u32,
    // },
    // Gather {
    //     comm: *const c_void,
    //     stream: *const c_void,
    //     sendcount: u64,
    //     datatype: u32,
    //     root: u32,
    // },
    // Scatter {
    //     comm: *const c_void,
    //     stream: *const c_void,
    //     recvcount: u64,
    //     datatype: u32,
    //     root: u32,
    // },
    Send {
        comm: *const c_void,
        stream: *const c_void,
        count: u64,
        datatype: u32,
        peer: u32,
    },
    Recv {
        comm: *const c_void,
        stream: *const c_void,
        count: u64,
        datatype: u32,
        peer: u32,
    },
}


fn get_nccl_op_name(op: u32) -> &'static str {
    match op {
        0 => "ncclSum",
        1 => "ncclProd",
        2 => "ncclMax",
        3 => "ncclMin",
        4 => "ncclAvg",
        _ => "ncclUnknownOp",
    }
}

fn get_nccl_datatype_name(datatype: u32) -> &'static str {
    match datatype {
        0 => "ncclInt8",
        1 => "ncclChar",
        2 => "ncclUint8",
        3 => "ncclInt32",
        4 => "ncclUint32",
        5 => "ncclInt64",
        6 => "ncclUint64",
        7 => "ncclFloat16",
        8 => "ncclFloat32",
        9 => "ncclFloat64",
        10 => "ncclBfloat16",
        _ => "ncclUnknownType",
    }
}

fn format_stream_for_nccl(stream: *const c_void) -> String {
    match cuda_stream_get_id(stream) {
        Ok(id) => id.to_string(),
        Err(err) => {
            log::warn!(
                "failed to get stream id for NCCL stream {:p}: {}",
                stream,
                err
            );
            format!("{:p}", stream)
        }
    }
}

impl Display for NCCLCommunication {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let api = self.api_name();
        match self {
            NCCLCommunication::AllReduce { comm, stream, count, datatype, op } => {
                write!(
                    f,
                    "<NCCL Kernel: {}(comm={:p}, count={}, datatype={}, op={}) on stream {}>",
                    api,
                    comm,
                    count,
                    get_nccl_datatype_name(*datatype),
                    get_nccl_op_name(*op),
                    format_stream_for_nccl(*stream),
                )
            }
            NCCLCommunication::Broadcast { comm, stream, count, datatype, root } => {
                write!(
                    f,
                    "<NCCL Kernel: {}(comm={:p}, count={}, datatype={}, root={}) on stream {}>",
                    api,
                    comm,
                    count,
                    get_nccl_datatype_name(*datatype),
                    root,
                    format_stream_for_nccl(*stream),
                )
            }
            NCCLCommunication::Reduce { comm, stream, count, datatype, op, root } => {
                write!(
                    f,
                    "<NCCL Kernel: {}(comm={:p}, count={}, datatype={}, op={}, root={}) on stream {}>",
                    api,
                    comm,
                    count,
                    get_nccl_datatype_name(*datatype),
                    get_nccl_op_name(*op),
                    root,
                    format_stream_for_nccl(*stream),
                )
            }
            NCCLCommunication::AllGather { comm, stream, sendcount, datatype } => {
                write!(
                    f,
                    "<NCCL Kernel: {}(comm={:p}, sendcount={}, datatype={}) on stream {}>",
                    api,
                    comm,
                    sendcount,
                    get_nccl_datatype_name(*datatype),
                    format_stream_for_nccl(*stream),
                )
            }
            NCCLCommunication::ReduceScatter { comm, stream, recvcount, datatype, op } => {
                write!(
                    f,
                    "<NCCL Kernel: {}(comm={:p}, recvcount={}, datatype={}, op={}) on stream {}>",
                    api,
                    comm,
                    recvcount,
                    get_nccl_datatype_name(*datatype),
                    get_nccl_op_name(*op),
                    format_stream_for_nccl(*stream),
                )
            }
            // NCCLCommunication::AlltoAll { comm, stream, count, datatype } => {
            //     write!(f, "<NCCL Kernel: {}(comm={:p}, count={}, datatype={}) on stream {}>",
            //            api, comm, count, get_nccl_datatype_name(*datatype), format_stream_for_nccl(*stream))
            // }
            // NCCLCommunication::Gather { comm, stream, sendcount, datatype, root } => {
            //     write!(f, "<NCCL Kernel: {}(comm={:p}, sendcount={}, datatype={}, root={}) on stream {}>",
            //            api, comm, sendcount, get_nccl_datatype_name(*datatype), root, format_stream_for_nccl(*stream))
            // }
            // NCCLCommunication::Scatter { comm, stream, recvcount, datatype, root } => {
            //     write!(f, "<NCCL Kernel: {}(comm={:p}, recvcount={}, datatype={}, root={}) on stream {}>",
            //            api, comm, recvcount, get_nccl_datatype_name(*datatype), root, format_stream_for_nccl(*stream))
            // }
            NCCLCommunication::Send { comm, stream, count, datatype, peer } => {
                write!(
                    f,
                    "<NCCL Kernel: {}(comm={:p}, count={}, datatype={}, peer={}) on stream {}>",
                    api,
                    comm,
                    count,
                    get_nccl_datatype_name(*datatype),
                    peer,
                    format_stream_for_nccl(*stream),
                )
            }
            NCCLCommunication::Recv { comm, stream, count, datatype, peer } => {
                write!(
                    f,
                    "<NCCL Kernel: {}(comm={:p}, count={}, datatype={}, peer={}) on stream {}>",
                    api,
                    comm,
                    count,
                    get_nccl_datatype_name(*datatype),
                    peer,
                    format_stream_for_nccl(*stream),
                )
            }
        }
    }
}

impl NCCLCommunication {
    pub fn stream(&self) -> *const c_void {
        match self {
            NCCLCommunication::AllReduce { stream, .. } => *stream,
            NCCLCommunication::Broadcast { stream, .. } => *stream,
            NCCLCommunication::Reduce { stream, .. } => *stream,
            NCCLCommunication::AllGather { stream, .. } => *stream,
            NCCLCommunication::ReduceScatter { stream, .. } => *stream,
            // NCCLCommunication::AlltoAll { stream, .. } => *stream,
            // NCCLCommunication::Gather { stream, .. } => *stream,
            // NCCLCommunication::Scatter { stream, .. } => *stream,
            NCCLCommunication::Send { stream, .. } => *stream,
            NCCLCommunication::Recv { stream, .. } => *stream,
        }
    }

    pub fn api_name(&self) -> &'static str {
        match self {
            NCCLCommunication::AllReduce { .. } => "ncclAllReduce",
            NCCLCommunication::Broadcast { .. } => "ncclBroadcast",
            NCCLCommunication::Reduce { .. } => "ncclReduce",
            NCCLCommunication::AllGather { .. } => "ncclAllGather",
            NCCLCommunication::ReduceScatter { .. } => "ncclReduceScatter",
            // NCCLCommunication::AlltoAll { .. } => "ncclAllToAll",
            // NCCLCommunication::Gather { .. } => "ncclGather",
            // NCCLCommunication::Scatter { .. } => "ncclScatter",
            NCCLCommunication::Send { .. } => "ncclSend",
            NCCLCommunication::Recv { .. } => "ncclRecv",
        }
    }
}