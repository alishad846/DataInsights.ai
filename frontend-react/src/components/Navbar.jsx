export default function Navbar({ title }) {
    const logout = () => {
        sessionStorage.removeItem("token");
        sessionStorage.removeItem("role");
        sessionStorage.removeItem("userName");
        window.location.href = "/login";
    };

    return (
        <div style={styles.nav}>
            <h3>{title}</h3>
            <button onClick={logout}>Logout</button>
        </div>
    );
}

const styles = {
    nav: {
        display: "flex",
        justifyContent: "space-between",
        padding: "15px",
        background: "#222",
        color: "#fff"
    }
};
