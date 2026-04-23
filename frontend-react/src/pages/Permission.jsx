import Navbar from "../components/Navbar";
import Sidebar from "../components/Sidebar";
import axios from "axios";
import { useEffect, useState } from "react";

export default function PermissionPage() {
    const [data, setData] = useState([]);

    const fetchData = () => {
        axios.get("http://localhost:5000/admin/permissions", {
            headers: { authorization: sessionStorage.getItem("token") }
        }).then(res => setData(res.data));
    };

    useEffect(() => {
        fetchData();
    }, []);

    const update = async (id, status) => {
        await axios.post(`http://localhost:5000/admin/permission/${id}`,
            { status },
            { headers: { authorization: sessionStorage.getItem("token") } }
        );
        fetchData();
    };

    return (
        <div>
            <Navbar title="Permissions" />
            <Sidebar />

            <div style={{ marginLeft: "220px", padding: "20px" }}>
                <table border="1" width="100%">
                    <thead>
                        <tr>
                            <th>Employee</th>
                            <th>Request</th>
                            <th>Status</th>
                            <th>Action</th>
                        </tr>
                    </thead>

                    <tbody>
                        {data.map(p => (
                            <tr key={p._id}>
                                <td>{p.employeeId?.name}</td>
                                <td>{p.requestType}</td>
                                <td>{p.status}</td>
                                <td>
                                    <button onClick={() => update(p._id, "accepted")}>Accept</button>
                                    <button onClick={() => update(p._id, "rejected")}>Reject</button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
