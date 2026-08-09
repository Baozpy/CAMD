public class SafeStringExample {

    public int getUsernameLength(String username) {
        if (username == null) {
            return 0;
        }

        return username.length();
    }

}